# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""HDF5 network exchange module."""

from typing import List, Tuple, Union
from lava.magma.core.decorator import implements
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
import numpy as np
import h5py

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.proc.lif.process import LIF
from lava.proc.sdn.process import Sigma, Delta, SigmaDelta
from lava.lib.dl.netx.utils import NetDict
from lava.lib.dl.netx.utils import optimize_weight_bits
from lava.lib.dl.netx.blocks.process import Input, Dense, Conv
from lava.lib.dl.netx.blocks.models import AbstractPyBlockModel


class Network(AbstractProcess):
    """Generates a Lava process for the network described in hdf5 config.

    Parameters
    ----------
    net_config : str
        name of the hdf5 config filename.
    num_layers : int, optional
        number of blocks to generate. An integer valuew will only generate the
        first ``num_layers`` blocks in the description. The actual number of
        generated layers may be less than ``num_layers``. If it is None, all
        the layers are generated. Defaults to None.
    has_graded_input : bool, optional
        flag indicating the input spike type of input layer. Defaults to False.
    """
    def __init__(
        self,
        **kwargs: Union[int, Tuple[int, ...]]
    ) -> None:
        super().__init__(**kwargs)
        self.filename = kwargs.pop('net_config')
        self.net_config = NetDict(self.filename)

        self.num_layers = kwargs.pop('num_layers', None)
        self.has_graded_input = kwargs.pop('has_graded_input', False)

        self.net_str = ''
        self.layers = self._create()

        self.in_layer = self.layers[0]
        self.out_layer = self.layers[-1]

        self.inp = InPort(shape=self.in_layer.inp.shape)
        self.out = OutPort(shape=self.out_layer.out.shape)

        self.inp.connect(self.in_layer.inp)
        self.out_layer.out.connect(self.out)

        self.has_graded_input = self.in_layer.has_graded_input
        self.has_graded_output = self.out_layer.has_graded_output

    def __str__(self) -> str:
        """Network description string."""
        return self.net_str

    def __len__(self) -> int:
        """Number of layers in the network."""
        return len(self.layers)

    @staticmethod
    def get_neuron_params(
        neuron_config: h5py.Group,
        input: bool = False,
    ) -> AbstractProcess:
        """Provides the correct neuron configuration process and parameters
        from the neuron description in hdf5 config.

        Parameters
        ----------
        neuron_config: h5py.Group
            hdf5 object describing the neuron configuration
        input: bool
            flag to indicate if the layer is input. For some cases special
            processing may be done.

        Returns
        -------
        AbstractProcess
            The Lava process that implements the neuron described.
        """
        neuron_type = neuron_config['type']
        if neuron_type in ['LOIHI', 'CUBA']:
            neuron_process = LIF
            neuron_params = {
                'neuron_proc': neuron_process,
                'vth': neuron_config['vThMant'],
                'du': neuron_config['iDecay'] - 1,
                'dv': neuron_config['vDecay'],
                'bias_exp': 6,
                'use_graded_spikes': False,
            }
            return neuron_params
        elif neuron_type in ['SDNN']:
            if input is True:
                # If it is an input layer (input is true) use delta process.
                neuron_process = Delta
                neuron_params = {
                    'neuron_proc': neuron_process,
                    'vth': neuron_config['vThMant'],
                    'state_exp': 6,
                    'wgt_exp': 6,
                    'use_graded_spikes': True,
                }
            elif 'sigma_output' in neuron_config.keys():
                neuron_process = Sigma
                neuron_params = {
                    'neuron_proc': neuron_process,
                    'use_graded_spikes': True,
                }
            else:
                neuron_process = SigmaDelta
                neuron_params = {
                    'neuron_proc': neuron_process,
                    'vth': neuron_config['vThMant'],
                    'state_exp': 6,
                    'wgt_exp': 6,
                    'use_graded_spikes': True,
                }
            return neuron_params

    @staticmethod
    def _table_str(
        type_str: str = '',
        width: Union[int, None] = None,
        height: Union[int, None] = None,
        channel: Union[int, None] = None,
        kernel: Union[int, Tuple[int, int], None] = None,
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int], None] = None,
        dilation: Union[int, Tuple[int, int], None] = None,
        groups: Union[int, None] = None,
        delay: bool = False,
        header: bool = False,
    ) -> str:
        """Helper function to print mapping output configuration
        for a layer/block."""
        if header is True:
            return '|   Type   |  W  |  H  |  C  '\
                '| ker | str | pad | dil | grp |delay|'

        else:
            def array_entry(data: Union[int, Tuple[int, int], None]) -> str:
                if data is None:
                    return f'{"":5s}|'
                elif len(data.shape) == 1:
                    return f'{data[0]:-2d},{data[1]:-2d}|'
                else:
                    return f'{data:-5d}|'

            entry = '|'
            entry += '{:10s}|'.format(type_str)
            entry += '{:-5d}|'.format(width)
            entry += '{:-5d}|'.format(height)
            entry += '{:-5d}|'.format(channel)
            entry += array_entry(kernel)
            entry += array_entry(stride)
            entry += array_entry(padding)
            entry += array_entry(dilation)
            entry += array_entry(groups)
            entry += '{:5s}|'.format(str(delay))

            return entry

    @staticmethod
    def create_input(layer_config: h5py.Group) -> Tuple[Input, str]:
        """Creates input layer from layer configuration.

        Parameters
        ----------
        layer_config : h5py.Group
            hdf5 handle to layer description.

        Returns
        -------
        AbstractProcess
            input block process.
        str
            table entry string for process.
        """
        shape = tuple(layer_config['shape'][::-1])  # WHC (XYZ)
        neuron_params = Network.get_neuron_params(
            layer_config['neuron'], input=True
        )

        if 'weight' in layer_config.keys():
            weight = int(layer_config['weight'])
        else:
            weight = 64

        if 'bias' in layer_config.keys():
            bias = int(layer_config['bias'])
        else:
            bias = 0

        transform = {'weight': weight, 'bias': bias}

        params = {  # arguments for Input block
            'shape': shape,
            'neuron_params': neuron_params,
            'transform': transform,
        }

        table_entry = Network._table_str(
            type_str='Input',
            width=shape[0], height=shape[1], channel=shape[2],
        )

        return Input(**params), table_entry

    @staticmethod
    def create_dense(
        layer_config: h5py.Group, has_graded_input: bool = False
    ) -> Tuple[Dense, str]:
        """Creates dense layer from layer configuration

        Parameters
        ----------
        layer_config : h5py.Group
            hdf5 handle to layer description.
        has_graded_input : bool, optional
            flag to indicate graded spikes at input, by default False.

        Returns
        -------
        AbstractProcess
            dense block process.
        str
            table entry string for process.
        """
        shape = (np.prod(layer_config['shape']),)
        neuron_params = Network.get_neuron_params(layer_config['neuron'])
        weight = layer_config['weight']
        if weight.ndim == 1:
            weight = weight.reshape(shape[0], -1)

        opt_weights = optimize_weight_bits(weight)
        weight, num_weight_bits, weight_exponent, sign_mode = opt_weights

        params = {  # arguments for dense block
            'shape': shape,
            'neuron_params': neuron_params,
            'weight': weight,
            'num_weight_bits': num_weight_bits,
            'weight_exponent': weight_exponent,
            'sign_mode': sign_mode,
            'has_graded_input': has_graded_input,
        }

        # optional arguments
        if 'bias' in layer_config.keys():
            params['bias'] = layer_config['bias']
        if 'delay' not in layer_config.keys():
            params['neuron_params']['delay_bits'] = 1
        else:
            pass  # TODO: set appropriate delay bits for synaptic delay

        table_entry = Network._table_str(
            type_str='Dense',
            width=1, height=1, channel=shape[0],
            delay='delay' in layer_config.keys(),
        )

        return Dense(**params), table_entry

    @staticmethod
    def create_conv(
        layer_config: h5py.Group,
        input_shape: Tuple[int, int, int],
        has_graded_input: bool = False
    ) -> Tuple[Conv, str]:
        """Creates conv layer from layer configuration

        Parameters
        ----------
        layer_config : h5py.Group
            hdf5 handle to layer description.
        input_shape : tuple of 3 ints
            shape of input to the block.
        has_graded_input : bool, optional
            flag to indicate graded spikes at input, by default False.

        Returns
        -------
        AbstractProcess
            dense block process.
        str
            table entry string for process.
        """
        shape = tuple(layer_config['shape'][::-1])  # WHC (XYZ)
        neuron_params = Network.get_neuron_params(layer_config['neuron'])
        weight = layer_config['weight'][:, :, ::-1, ::-1]
        weight = weight.reshape(weight.shape[:4]).transpose((0, 3, 2, 1))
        stride = layer_config['stride'][::-1]
        padding = layer_config['padding'][::-1]
        dilation = layer_config['dilation'][::-1]
        groups = layer_config['groups']

        params = {  # arguments for conv block
            'input_shape': input_shape,
            'shape': shape,
            'neuron_params': neuron_params,
            'weight': weight,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'has_graded_input': has_graded_input,
        }

        # Optional arguments
        if 'bias' in layer_config.keys():
            params['bias'] = layer_config['bias']

        if 'delay' not in layer_config.keys():
            params['neuron_params']['delay_bits'] = 1
        else:
            pass

        table_entry = Network._table_str(
            type_str='Conv',
            width=shape[0], height=shape[1], channel=shape[2],
            kernel=np.array([weight.shape[i] for i in [1, 2]]),
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            delay='delay' in layer_config.keys(),
        )

        return Conv(**params), table_entry

    @staticmethod
    def create_pool(layer_config: h5py.Group) -> None:
        raise NotImplementedError

    @staticmethod
    def create_convT(layer_config: h5py.Group) -> None:
        raise NotImplementedError

    @staticmethod
    def create_unpool(layer_config: h5py.Group) -> None:
        raise NotImplementedError

    @staticmethod
    def create_average(layer_config: h5py.Group) -> None:
        raise NotImplementedError

    @staticmethod
    def create_concat(layer_config: h5py.Group) -> None:
        raise NotImplementedError

    def _create(self) -> List[AbstractProcess]:
        has_graded_input_next = self.has_graded_input
        flatten_next = False
        layers = []
        layer_config = self.net_config['layer']

        self.net_str += self._table_str(header=True) + '\n'

        num_layers = len(layer_config)
        if self.num_layers is not None:
            num_layers = min(num_layers, self.num_layers)

        for i in range(num_layers):
            layer_type = layer_config[i]['type']

            if layer_type == 'input':
                layer, table = self.create_input(layer_config[i])
                layers.append(layer)
                has_graded_input_next = layer.has_graded_output

            elif layer_type == 'conv':
                layer, table = self.create_conv(
                    layer_config=layer_config[i],
                    input_shape=layers[-1].shape,
                    has_graded_input=has_graded_input_next
                )
                layers.append(layer)
                has_graded_input_next = layer.has_graded_output
                if len(layers) > 1:
                    layers[-2].out.connect(layers[-1].inp)

            elif layer_type == 'pool':
                raise NotImplementedError(f'{layer_type} is not implemented.')

            elif layer_type == 'convT':
                raise NotImplementedError(f'{layer_type} is not implemented.')

            elif layer_type == 'unpool':
                raise NotImplementedError(f'{layer_type} is not implemented.')

            elif layer_type == 'flatten':
                flatten_next = True
                table = None

            elif layer_type == 'dense':
                layer, table = self.create_dense(
                    layer_config=layer_config[i],
                    has_graded_input=has_graded_input_next
                )
                layers.append(layer)
                has_graded_input_next = layer.has_graded_output
                if flatten_next:
                    layers[-2].out.transpose([2, 1, 0]).flatten().connect(
                        layers[-1].inp
                    )
                    flatten_next = False
                else:
                    if len(layers) > 1:
                        layers[-2].out.connect(layers[-1].inp)

            elif layer_type == 'average':
                raise NotImplementedError(f'{layer_type} is not implemented.')

            elif layer_type == 'concat':
                raise NotImplementedError(f'{layer_type} is not implemented.')

            if table:
                self.net_str += table + '\n'

        self.net_str = self.net_str[:-1]
        return layers


@implements(proc=Network, protocol=LoihiProtocol)
class PyNetworkModel(AbstractPyBlockModel):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)
