# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Sigma Delta layer blocks."""

import numpy as np
import torch

from . import base
from ..neuron import sigma_delta
from ..synapse import layer as synapse
from ..axon import Delay


class AbstractSDRelu(torch.nn.Module):
    """Abstract Sigma Delta block class. This should never be instantiated on
    it's own."""
    def __init__(self, *args, **kwargs):
        super(AbstractSDRelu, self).__init__(*args, **kwargs)
        if self.neuron_params is not None:
            self.neuron = sigma_delta.Neuron(**self.neuron_params)
        delay = kwargs['delay'] if 'delay' in kwargs.keys() else False
        self.delay = Delay(max_delay=62) if delay is True else None
        # Disable delay shift for SDNN
        if 'delay_shift' in kwargs.keys():
            self.delay_shift = kwargs['delay_shift']
        else:
            self.delay_shift = False
        del self.neuron_params


def _doc_from_base(base_doc):
    return base_doc.__doc__.replace(
        'Abstract', 'Sigma Delta'
    ).replace(
        'neuron parameter', 'Sigma Delta neuron parameter'
    ).replace(
        'This should never be instantiated on its own.',
        'The block is 8 bit quantization ready.'
    )


class Input(AbstractSDRelu, base.AbstractInput):
    def __init__(self, *args, **kwargs):
        super(Input, self).__init__(*args, **kwargs)
        if self.neuron is not None:
            self.pre_hook_fx = self.neuron.quantize_8bit
        # del self.synapse

    def forward(self, x):
        """
        """
        if self.neuron is not None:
            z = self.pre_hook_fx(x)
            if self.weight is not None:
                z = z * self.pre_hook_fx(self.weight)
            if self.bias is not None:
                z = z + self.pre_hook_fx(self.bias)
            x = self.neuron.delta(z)

        if self.input_shape is None:
            self.input_shape = self.neuron.delta.shape
            self.neuron.sigma.shape = self.neuron.delta.shape
            self.neuron.shape = self.neuron.delta.shape

        return x


Input.__doc__ = _doc_from_base(base.AbstractInput)


class Flatten(base.AbstractFlatten):
    def __init__(self, *args, **kwargs):
        super(Flatten, self).__init__(*args, **kwargs)


Flatten.__doc__ = _doc_from_base(base.AbstractFlatten)


class Average(base.AbstractAverage):
    def __init__(self, *args, **kwargs):
        super(Average, self).__init__(*args, **kwargs)


Average.__doc__ = _doc_from_base(base.AbstractAverage)


class Dense(AbstractSDRelu, base.AbstractDense):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.synapse = synapse.Dense(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Dense.__doc__ = _doc_from_base(base.AbstractDense)


class Conv(AbstractSDRelu, base.AbstractConv):
    def __init__(self, *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        self.synapse = synapse.Conv(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Conv.__doc__ = _doc_from_base(base.AbstractConv)


class ConvT(AbstractSDRelu, base.AbstractConvT):
    def __init__(self, *args, **kwargs):
        super(ConvT, self).__init__(*args, **kwargs)
        self.synapse = synapse.ConvTranspose(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


ConvT.__doc__ = _doc_from_base(base.AbstractConvT)


class Pool(AbstractSDRelu, base.AbstractPool):
    def __init__(self, *args, **kwargs):
        super(Pool, self).__init__(*args, **kwargs)
        self.synapse = synapse.Pool(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Pool.__doc__ = _doc_from_base(base.AbstractPool)


class Unpool(AbstractSDRelu, base.AbstractUnpool):
    def __init__(self, *args, **kwargs):
        super(Unpool, self).__init__(*args, **kwargs)
        self.synapse = synapse.Unpool(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Unpool.__doc__ = _doc_from_base(base.AbstractUnpool)


# class KWTA(AbstractSDRelu, base.AbstractKWTA):
#     def __init__(self, *args, **kwargs):
#         super(KWTA, self).__init__(*args, **kwargs)
#         self.synapse = synapse.Dense(**self.synapse_params)
#         if 'pre_hook_fx' not in kwargs.keys():
#             self.synapse.pre_hook_fx = self.neuron.quantize_8bit
#         del self.synapse_params


# class Affine(AbstractSDRelu, base.AbstractAffine):
#     def __init__(self, *args, **kwargs):
#         super(Affine, self).__init__(*args, dynamics=False, **kwargs)
#         self.synapse = synapse.Dense(**self.synapse_params)
#         if 'pre_hook_fx' not in kwargs.keys():
#             self.synapse.pre_hook_fx = self.neuron.quantize_8bit
#         self.neuron.threshold = None
#         # this disables spike and reset in dynamics
#         del self.synapse_params


class Output(AbstractSDRelu):
    """Sigma Delta output block class. The block is 8 bit quantization ready.

    Parameters
    ----------
    neuron_params : dict
        a dictionary of sigma delta neuron parameters.
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, in_neurons, out_neurons,
        weight_scale=1, weight_norm=False,
        count_log=False,
    ):
        self.neuron_params = neuron_params
        super(Output, self).__init__()
        self.synapse = synapse.Dense(
            in_neurons, out_neurons,
            weight_scale=weight_scale,
            weight_norm=weight_norm,
            pre_hook_fx=self.neuron.quantize_8bit,
        )

        self.count_log = count_log

    def forward(self, x):
        """
        """
        x = self.neuron.sigma(self.synapse(x))

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.sigma.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:2]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['dense'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.sigma.shape))
        handle.create_dataset('inFeatures', data=self.synapse.in_channels)
        handle.create_dataset('outFeatures', data=self.synapse.out_channels)

        if hasattr(self.synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.synapse.imag)
            )
        else:
            handle.create_dataset('weight', data=weight(self.synapse))

        if self.delay is not None:
            handle.create_dataset('delay', data=delay(self.delay))

        device_params = self.neuron.device_params
        device_params['sigma_output'] = True
        for key, value in device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
