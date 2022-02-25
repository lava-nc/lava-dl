# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Auto network generation module form network description. We support hdf5
network description for now. It is intended to load a model and perform
fine tuning or use it as a pretrained feature extractor."""

import torch
import h5py
from . import neuron
from . import block


def get_classes(neuron_type=None):
    """Maps slayer class from neuron type.

    Parameters
    ----------
    neuron_type : string
        neuron type description. None means cuba neuron. Defaults to None.

    Returns
    -------
    neuron_class, block_class
        neuron class and block class.

    """
    if neuron_type is None or neuron_type == 'CUBA' or neuron_type == 'LOIHI':
        return neuron.cuba, block.cuba

    raise Exception(f'{neuron_type=} is not implemented.')


def get_neuron_params(neuron_handle, neuron_class):
    """Gets neuron parameters from the hdf5 description handle.

    Parameters
    ----------
    neuron_handle : hdf5 handle
        handle to hdf5 object that describes the neuron.
    neuron_class : slayer.neuron.*
        neuron class type

    Returns
    -------
    dict
        dictionary of neuron parameters.

    """
    neuron_params_dict = {}
    for key in neuron_handle.keys():
        neuron_params_dict[key] = neuron_handle[key][()]

    return neuron_class.neuron_params(neuron_params_dict)


class SequentialNetwork(torch.nn.Module):
    """Creates sequential network from hdf5 network description.

    Parameters
    ----------
    network_config : str
        name of network configuration description.
    persistent_state : bool
        flag for persistent state. Defaults to False.
    reduction : str or None
        Reduction of output spike. Options are 'sum' or 'mean'.
        None means no reduction. Defaults to None.
    weight_norm : bool
        flag to enable weight norm. Defaults to False.
    count_log : bool
        flag to enable count statistics. Defaults to False.

    Returns
    -------
    torch module
        network module.

    """
    # this creates network from hdf5 file
    # this is intended be used to load back the model and do small fine
    # tuning or inference only
    def __init__(
        self, network_config,
        persistent_state=False, reduction=None, weight_norm=False,
        count_log=False
    ):
        super(SequentialNetwork, self).__init__()

        self.handle = h5py.File(network_config, 'r')
        self.persistent_state = persistent_state
        self.reduction = reduction
        self.count_log = count_log
        self.weight_norm = weight_norm
        self.blocks = torch.nn.ModuleList([])

        for layer in range(len(self.handle['layer'])):
            self.read_block(self.handle['layer'][f'{layer}'])

        # TODO: handle non-shared parameters

    def input_block(self, layer_handle):
        """
        """
        shape = layer_handle['shape'][()]
        if 'neuron' in layer_handle.keys():
            neuron_handle = layer_handle['neuron']
            if 'type' in neuron_handle.keys():
                neuron_type = neuron_handle['type'][()].decode('utf-8')
            else:
                neuron_type = None
            neuron_class, block_class = get_classes(neuron_type)
            neuron_params = get_neuron_params(neuron_handle, neuron_class)
            neuron_params['persistent_state'] = self.persistent_state
            if 'weight' in layer_handle.keys():
                weight = layer_handle['weight'][()]
            else:
                weight = 1
            if 'bias' in layer_handle.keys():
                bias = layer_handle['bias'][()]
            else:
                bias = 0
            block = block_class.Input(
                neuron_params, weight, bias, count_log=self.count_log
            )
        else:
            neuron_class, block_class = get_classes()
            neuron_params = None
            block = block_class.Input(neuron_params, count_log=self.count_log)

        block.input_shape = torch.Size(shape)
        return block

    def flatten_block(self, layer_handle):
        """
        """
        shape = layer_handle['shape'][()]
        _, block_class = get_classes()
        block = block_class.Flatten(count_log=self.count_log)
        block.shape = torch.Size(shape)
        return block

    def average_block(self, layer_handle):
        """
        """
        shape = layer_handle['shape'][()]
        _, block_class = get_classes()
        block = block_class.Average(
            num_outputs=shape[0],
            count_log=self.count_log
        )
        return block

    def dense_block(self, layer_handle):
        """
        """
        shape = layer_handle['shape'][()]
        neuron_handle = layer_handle['neuron']
        if 'type' in neuron_handle.keys():
            neuron_type = neuron_handle['type'][()].decode('utf-8')
        else:
            neuron_type = None
        neuron_class, block_class = get_classes(neuron_type)
        neuron_params = get_neuron_params(neuron_handle, neuron_class)
        neuron_params['persistent_state'] = self.persistent_state

        weight = layer_handle['weight']
        if 'bias' in layer_handle.keys():
            bias = layer_handle['bias'][()]
        else:
            bias = None
        if 'delay' in layer_handle.keys():
            delay = layer_handle['delay'][()]
        else:
            delay = None

        block = block_class.Dense(
            neuron_params,
            in_neurons=int(layer_handle['inFeatures'][()]),
            out_neurons=int(layer_handle['outFeatures'][()]),
            delay=delay is not None,
            count_log=self.count_log
        )

        block.neuron.shape = torch.Size(shape)
        if neuron_type in ['rf']:  # complex neurons
            block.synapse.real.weight.data = torch.FloatTensor(
                weight['real'][()] / block.neuron.w_scale
            ).reshape(block.synapse.shape).to()
            block.synapse.imag.weight.data = torch.FloatTensor(
                weight['imag'][()] / block.neuron.w_scale
            ).reshape(block.synapse.shape)
        else:
            block.synapse.weight.data = torch.FloatTensor(
                weight[()] / block.neuron.w_scale
            ).reshape(block.synapse.shape)

        if bias is not None:
            block.bias = torch.FloatTensor(bias)

        if delay is not None:
            block.delay.delay.data = torch.FloatTensor(delay)
            block.delay.init = True

        if self.weight_norm is True:
            block.enable_weight_norm()

        return block

    def read_block(self, layer_handle):
        """
        """
        block_type = layer_handle['type'][()][0].decode('utf-8')
        if block_type == 'input':
            self.blocks.append(self.input_block(layer_handle))
        elif block_type == 'flatten':
            self.blocks.append(self.flatten_block(layer_handle))
        elif block_type == 'average':
            self.blocks.append(self.average_block(layer_handle))
        elif block_type == 'dense':
            self.blocks.append(self.dense_block(layer_handle))
        elif block_type == 'conv':
            self.blocks.append(self.conv_block(layer_handle))
        elif block_type == 'pool':
            self.blocks.append(self.pool_block(layer_handle))
        else:
            raise Exception(f'{block_type=} is not implemented.')

    def forward(self, spike):
        """
        """
        # count = []
        for blk in self.blocks:
            spike = blk(spike)
            # count.append(torch.mean(spike).item())

        if self.reduction == 'sum':
            spike = torch.sum(spike.reshape(spike.shape[0], 11, -1), dim=2)
        elif self.reduction == 'mean':
            spike = torch.mean(spike.reshape(spike.shape[0], 11, -1), dim=2)

        return spike
        # return (
        #     spike,
        #     torch.FloatTensor(count).reshape((1, -1)).to(spike.get_device()),
        # )
