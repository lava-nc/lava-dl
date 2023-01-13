# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""HDF5 network exchange module."""

from typing import List, Optional, Tuple, Union
import warnings
from lava.magma.core.decorator import implements
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.rf.process import RF
from lava.proc.rf_iz.process import RF_IZ
import numpy as np
import h5py

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.proc.lif.process import LIF, LIFReset
from lava.proc.sdn.process import Sigma, Delta, SigmaDelta
from lava.lib.dl.slayer.neuron.rf import neuron_params as get_rf_params
from lava.lib.dl.netx.utils import NetDict
from lava.lib.dl.netx.utils import optimize_weight_bits
from lava.lib.dl.netx.blocks.process import Input, Dense, Conv, ComplexDense
from lava.lib.dl.netx.blocks.models import AbstractPyBlockModel


class Network(AbstractProcess):
    """Generates a Lava process for the network described in hdf5 config.

    Parameters
    ----------
    net_config : str
        name of the hdf5 config filename.
    num_layers : int, optional
        number of blocks to generate. An integer value will only generate the
        first ``num_layers`` blocks in the description. The actual number of
        generated layers may be less than ``num_layers``. If it is None, all
        the layers are generated. Defaults to None.
    skip_layers : int
        number of blocks to skip. The nework generator will skip the first
        ``skip_layers`` blocks that have been generated accoridng to
        ``num_layers`` parametr. If it is None, no layers are skipped.
        Defaults to None.
    input_message_bits : int, optional
        number of message bits in input spike. Defaults to 0 meaning unary
        spike.
    input_shape : tuple of ints, optional
        shape of input to the network. If None, input layer is assumed to be
        the first layer. Defaults to None.
    reset_interval: int, optional
        determines how often the network needs to be reset. Reset is
        orchestrated with a timestep offset as per the layer number. If that is
        not desired, it can be changed by directly accessing the layer's
        neuron's reset parameter. None means no reset. Defaults to None.
    reset_offset: int
        determines the phase shift of network reset if enabled. Defaults to 0.
    """
    def __init__(self,
                 net_config: str,
                 num_layers: Optional[int] = None,
                 skip_layers: int = 0,
                 input_message_bits: Optional[int] = 0,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 reset_interval: Optional[int] = None,
                 reset_offset: int = 0) -> None:
        super().__init__(net_config=net_config,
                         num_layers=num_layers,
                         input_message_bits=input_message_bits)
        self.filename = net_config
        self.net_config = NetDict(self.filename)

        self.num_layers = num_layers
        self.skip_layers = skip_layers
        self.input_message_bits = input_message_bits
        self.input_shape = input_shape
        self.reset_interval = reset_interval
        self.reset_offset = reset_offset

        self.net_str = ''
        self.layers = self._create()

        self.in_layer = self.layers[0]
        self.out_layer = self.layers[-1]

        self.inp = InPort(shape=self.in_layer.inp.shape)
        self.out = OutPort(shape=self.out_layer.out.shape)

        self.inp.connect(self.in_layer.inp)
        self.out_layer.out.connect(self.out)

        self.output_message_bits = self.out_layer.output_message_bits

    def __str__(self) -> str:
        """Network description string."""
        return self.net_str

    def __len__(self) -> int:
        """Number of layers in the network."""
        return len(self.layers)

    @staticmethod
    def get_neuron_params(neuron_config: h5py.Group,
                          input: bool = False,
                          reset_interval: Optional[int] = None,
                          reset_offset: int = 0) -> AbstractProcess:
        """Provides the correct neuron configuration process and parameters
        from the neuron description in hdf5 config.

        Parameters
        ----------
        neuron_config: h5py.Group
            hdf5 object describing the neuron configuration
        input: bool
            flag to indicate if the layer is input. For some cases special
            processing may be done.
        reset_interval: int, optional
            the reset interval. None means no reset. Default is None.
        reset_offset: int
            the offset/phase of reset. It is only valid of reset_interval is
            not None.

        Returns
        -------
        AbstractProcess
            The Lava process that implements the neuron described.
        """
        neuron_type = neuron_config['type']
        num_message_bits = None
        if 'messageBits' in neuron_config.keys():
            num_message_bits = neuron_config['messageBits']
        if neuron_type in ['LOIHI', 'CUBA']:
            if num_message_bits is None:
                num_message_bits = 0  # default value
            if reset_interval is None:
                neuron_process = LIF
            else:
                neuron_process = LIFReset
            neuron_params = {'neuron_proc': neuron_process,
                             # Map bias parameter in hdf5 config to bias_mant
                             'bias_key': 'bias_mant',
                             # Rest of the neuron params
                             'vth': neuron_config['vThMant'],
                             'du': neuron_config['iDecay'] - 1,
                             'dv': neuron_config['vDecay'],
                             'bias_exp': 6,
                             'num_message_bits': num_message_bits}
            if reset_interval is not None:
                neuron_params['reset_interval'] = reset_interval
                neuron_params['reset_offset'] = reset_offset
            return neuron_params
        elif neuron_type in ['SDNN']:
            if reset_interval is not None:
                warnings.warn("Reset is not supported with Sigma Delta "
                              "neurons. It will be ignored.")
            if num_message_bits is None:
                num_message_bits = 16  # default value
            if input is True:
                # Use delta process.
                neuron_process = Delta
                neuron_params = {'neuron_proc': neuron_process,
                                 'vth': neuron_config['vThMant'],
                                 'spike_exp': 6,
                                 'state_exp': 6,
                                 'num_message_bits': num_message_bits}
            elif 'sigma_output' in neuron_config.keys():
                neuron_process = Sigma
                neuron_params = {'neuron_proc': neuron_process,
                                 'num_message_bits': num_message_bits}
            else:
                neuron_process = SigmaDelta
                neuron_params = {'neuron_proc': neuron_process,
                                 'vth': neuron_config['vThMant'],
                                 'spike_exp': 6,
                                 'state_exp': 6,
                                 'num_message_bits': num_message_bits}
            return neuron_params
        elif "RF" in neuron_type:
            if num_message_bits is None:
                num_message_bits = 0  # default value
            neuron_process = RF if "PHASE" in neuron_type else RF_IZ
            neuron_params = get_rf_params(neuron_config)
            neuron_params = {
                'neuron_proc': neuron_process,
                'vth': neuron_config['vThMant'],
                'period': neuron_params['period'],
                'alpha': neuron_params['decay'],
                'state_exp': 6,
                'decay_bits': 12
            }
            return neuron_params

    @staticmethod
    def _table_str(type_str: str = '',
                   width: Optional[int] = None,
                   height: Optional[int] = None,
                   channel: Optional[int] = None,
                   kernel: Optional[Union[int, Tuple[int, int]]] = None,
                   stride: Optional[Union[int, Tuple[int, int]]] = None,
                   padding: Optional[Union[int, Tuple[int, int]]] = None,
                   dilation: Optional[Union[int, Tuple[int, int]]] = None,
                   groups: Optional[int] = None,
                   delay: bool = False,
                   header: bool = False) -> str:
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
    def create_input(layer_config: h5py.Group,
                     reset_interval: Optional[int] = None,
                     reset_offset: int = 0) -> Tuple[Input, str]:
        """Creates input layer from layer configuration.

        Parameters
        ----------
        layer_config : h5py.Group
            hdf5 handle to layer description.
        reset_interval: int, optional
            the reset interval. None means no reset. Default is None.
        reset_offset: int
            the offset/phase of reset. It is only valid of reset_interval is
            not None.

        Returns
        -------
        AbstractProcess
            input block process.
        str
            table entry string for process.
        """
        shape = tuple(layer_config['shape'][::-1])  # WHC (XYZ)
        neuron_params = Network.get_neuron_params(layer_config['neuron'],
                                                  reset_interval=reset_interval,
                                                  reset_offset=reset_offset,
                                                  input=True)

        if 'weight' in layer_config.keys():
            weight = int(layer_config['weight'])
        else:
            weight = 64

        if 'bias' in layer_config.keys():
            bias = int(layer_config['bias'])
        else:
            bias = 0

        transform = {'weight': weight, 'bias': bias}

        # arguments for Input block
        params = {'shape': shape,
                  'neuron_params': neuron_params,
                  'transform': transform}

        if len(shape) == 1:
            table_entry = Network._table_str(type_str='Input',
                                             width=1,
                                             height=1,
                                             channel=shape[0])
        elif len(shape) == 2:
            table_entry = Network._table_str(type_str='Input',
                                             width=shape[0],
                                             height=shape[1],
                                             channel=1)
        else:
            table_entry = Network._table_str(type_str='Input',
                                             width=shape[0],
                                             height=shape[1],
                                             channel=shape[2])

        return Input(**params), table_entry

    @staticmethod
    def create_dense(layer_config: h5py.Group,
                     input_message_bits: int = 0,
                     reset_interval: Optional[int] = None,
                     reset_offset: int = 0) -> Tuple[Dense, str]:
        """Creates dense layer from layer configuration

        Parameters
        ----------
        layer_config : h5py.Group
            hdf5 handle to layer description.
        input_message_bits : int, optional
            number of message bits in input spike. Defaults to 0 meaning unary
            spike.
        reset_interval: int, optional
            the reset interval. None means no reset. Default is None.
        reset_offset: int
            the offset/phase of reset. It is only valid of reset_interval is
            not None.

        Returns
        -------
        AbstractProcess
            dense block process.
        str
            table entry string for process.
        """
        shape = (np.prod(layer_config['shape']),)

        if 'neuron' in layer_config.keys():
            neuron_config = layer_config['neuron']
        else:
            # Non-leaky integrator by default
            neuron_config = {'type' : 'CUBA',
                             'iDecay' : 0,
                             'vDecay' : 4096,
                             'vThMant' : 1 << 18 - 1,
                             'refDelay' : 1,
                             'gradedSpike' : False}
        neuron_params = Network.get_neuron_params(neuron_config,
                                                  reset_interval=reset_interval,
                                                  reset_offset=reset_offset)
        if "weight/imag" in layer_config.f:
            weight_real = layer_config['weight/real']
            weight_imag = layer_config['weight/imag']
            if weight_real.ndim == 1:
                weight_real = weight_real.reshape(shape[0], -1)
                weight_imag = weight_imag.reshape(shape[0], -1)

            opt_weights_real = optimize_weight_bits(weight_real)
            opt_weights_imag = optimize_weight_bits(weight_imag)
            weight_real, num_weight_bits_real, weight_exponent_real,\
                sign_mode_real = opt_weights_real
            weight_imag, num_weight_bits_imag, weight_exponent_imag,\
                sign_mode_imag = opt_weights_imag

            # arguments for complex dense block
            params = {'shape': shape,
                      'neuron_params': neuron_params,
                      'weight_real': weight_real,
                      'weight_imag': weight_imag,
                      'num_weight_bits_real': num_weight_bits_real,
                      'num_weight_bits_imag': num_weight_bits_imag,
                      'weight_exponent_real': weight_exponent_real,
                      'weight_exponent_imag': weight_exponent_imag,
                      'sign_mode_real': sign_mode_real,
                      'sign_mode_imag': sign_mode_imag,
                      'input_message_bits': input_message_bits}

            proc = ComplexDense(**params)

        else:
            weight = layer_config['weight']
            if weight.ndim == 1:
                weight = weight.reshape(shape[0], -1)

            opt_weights = optimize_weight_bits(weight)
            weight, num_weight_bits, weight_exponent, sign_mode = opt_weights

            # arguments for dense block
            params = {'shape': shape,
                      'neuron_params': neuron_params,
                      'weight': weight,
                      'num_weight_bits': num_weight_bits,
                      'weight_exponent': weight_exponent,
                      'sign_mode': sign_mode,
                      'input_message_bits': input_message_bits}

            # optional arguments
            if 'bias' in layer_config.keys():
                params['bias'] = layer_config['bias']

            proc = Dense(**params)
        table_entry = Network._table_str(type_str='Dense', width=1, height=1,
                                         channel=shape[0],
                                         delay='delay' in layer_config.keys())
        return proc, table_entry

    @staticmethod
    def create_conv(layer_config: h5py.Group,
                    input_shape: Tuple[int, int, int],
                    input_message_bits: int = 0,
                    reset_interval: Optional[int] = None,
                    reset_offset: int = 0) -> Tuple[Conv, str]:
        """Creates conv layer from layer configuration

        Parameters
        ----------
        layer_config : h5py.Group
            hdf5 handle to layer description.
        input_shape : tuple of 3 ints
            shape of input to the block.
        input_message_bits : int, optional
            number of message bits in input spike. Defaults to 0 meaning unary
            spike.
        reset_interval: int, optional
            the reset interval. None means no reset. Default is None.
        reset_offset: int
            the offset/phase of reset. It is only valid of reset_interval is
            not None.

        Returns
        -------
        AbstractProcess
            dense block process.
        str
            table entry string for process.
        """
        def expand(x):
            if np.isscalar(x):
                return np.array([x, x])
            return x[::-1]
        shape = tuple(layer_config['shape'][::-1])  # WHC (XYZ)
        neuron_params = Network.get_neuron_params(layer_config['neuron'],
                                                  reset_interval=reset_interval,
                                                  reset_offset=reset_offset)
        weight = layer_config['weight'][:, :, ::-1, ::-1]
        weight = weight.reshape(weight.shape[:4]).transpose((0, 3, 2, 1))
        stride = expand(layer_config['stride'])
        padding = expand(layer_config['padding'])
        dilation = expand(layer_config['dilation'])
        groups = layer_config['groups']

        # arguments for conv block
        params = {'input_shape': input_shape,
                  'shape': shape,
                  'neuron_params': neuron_params,
                  'weight': weight,
                  'stride': stride,
                  'padding': padding,
                  'dilation': dilation,
                  'groups': groups,
                  'input_message_bits': input_message_bits}

        # Optional arguments
        if 'bias' in layer_config.keys():
            params['bias'] = layer_config['bias']

        # if 'delay' not in layer_config.keys():
        #     params['neuron_params']['delay_bits'] = 1
        # else:
        #     pass

        kernel = np.array([weight.shape[i] for i in [1, 2]])
        table_entry = Network._table_str(type_str='Conv',
                                         width=shape[0],
                                         height=shape[1],
                                         channel=shape[2],
                                         kernel=kernel,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         delay='delay' in layer_config.keys())

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
        input_message_bits = self.input_message_bits
        flatten_next = False
        layers = []
        layer_config = self.net_config['layer']

        self.net_str += self._table_str(header=True) + '\n'

        num_layers = len(layer_config)
        if self.num_layers is not None:
            num_layers = min(num_layers, self.num_layers)

        reset_interval = self.reset_interval
        reset_offset = self.reset_offset + 1  # time starts from 1 in hardware
        for i in range(num_layers):
            layer_type = layer_config[i]['type']
            if layer_type == 'input':
                table = None
                if 'neuron' in layer_config[i].keys():
                    layer, table = self.create_input(
                        layer_config[i],
                        reset_interval=reset_interval,
                        reset_offset=reset_offset)
                    if i >= self.skip_layers:
                        layers.append(layer)
                        reset_offset += 1
                    else:
                        self.input_shape = layer.shape
                    input_message_bits = layer.output_message_bits
                elif 'shape' in layer_config[i].keys():
                    self.input_shape = tuple(layer_config[i]['shape'][::-1])

            elif layer_type == 'conv':
                if len(layers) > 0:
                    input_shape = layers[-1].shape
                elif self.input_shape:
                    input_shape = self.input_shape
                else:
                    raise RuntimeError('Input shape could not be inferred. '
                                       'Try explicitly specifying input_shape '
                                       'in hdf5.Network(...) in (x, y, f) '
                                       'order')
                layer, table = self.create_conv(
                    layer_config=layer_config[i],
                    input_shape=input_shape,
                    input_message_bits=input_message_bits,
                    reset_interval=reset_interval,
                    reset_offset=reset_offset)
                if i >= self.skip_layers:
                    layers.append(layer)
                    reset_offset += 1
                else:
                    self.input_shape = layer.shape
                input_message_bits = layer.output_message_bits
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
                    input_message_bits=input_message_bits,
                    reset_interval=reset_interval,
                    reset_offset=reset_offset)
                if i >= self.skip_layers:
                    layers.append(layer)
                    reset_offset += 1
                input_message_bits = layer.output_message_bits
                if flatten_next:
                    layers[-2].out.transpose([2, 1, 0]).flatten().connect(
                        layers[-1].inp
                    )
                    flatten_next = False
                else:
                    if len(layers) > 1:
                        layers[-2].out.connect(layers[-1].inp)
            elif layer_type == "dense_comp":
                layer, table = self.create_complex_dense(
                    layer_config=layer_config[i],
                    input_message_bits=input_message_bits
                )
                layers.append(layer)
                input_message_bits = layer.output_message_bits
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

            if table and i >= self.skip_layers:
                self.net_str += table + '\n'

        self.net_str = self.net_str[:-1]
        return layers


@implements(proc=Network, protocol=LoihiProtocol)
class PyNetworkModel(AbstractPyBlockModel):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)
