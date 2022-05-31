# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Resonate and Fire - Izhikevich layer blocks."""

import torch

from . import base
from ..neuron import rf_iz
from ..synapse import complex as synapse
from ..axon import Delay


class AbstractRFIz(torch.nn.Module):
    """Abstract Resonate and Fire - Izhikevich block class. This should never
    be instantiated on it's own."""
    def __init__(self, *args, **kwargs):
        super(AbstractRFIz, self).__init__(*args, **kwargs)
        if self.neuron_params is not None:
            self.neuron = rf_iz.Neuron(**self.neuron_params)
        delay = kwargs['delay'] if 'delay' in kwargs.keys() else False
        self.delay = Delay(max_delay=62) if delay is True else None
        del self.neuron_params


def _doc_from_base(base_doc):
    """ """
    return base_doc.__doc__.replace(
        'Abstract', 'Resonate & Fire Izhikevich'
    ).replace(
        'neuron parameter', 'RF-Izhikevich neuron parameter'
    ).replace(
        'This should never be instantiated on its own.',
        'The block is 8 bit quantization ready.'
    )


class Input(AbstractRFIz, base.AbstractInput):
    def __init__(self, *args, **kwargs):
        super(Input, self).__init__(*args, **kwargs)
        if self.neuron is not None:
            self.pre_hook_fx = self.neuron.quantize_8bit


Input.__doc__ = _doc_from_base(base.AbstractInput)


class Flatten(base.AbstractFlatten):
    def __init__(self, *args, **kwargs):
        super(Flatten, self).__init__(*args, **kwargs)


Flatten.__doc__ = _doc_from_base(base.AbstractFlatten)


class Average(base.AbstractAverage):
    def __init__(self, *args, **kwargs):
        super(Average, self).__init__(*args, **kwargs)


Average.__doc__ = _doc_from_base(base.AbstractAverage)


class Affine(AbstractRFIz, base.AbstractAffine):
    def __init__(self, *args, **kwargs):
        super(Affine, self).__init__(*args, **kwargs)
        self.synapse = synapse.Dense(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        self.neuron.threshold = None
        # this disables spike and reset in dynamics
        del self.synapse_params


Affine.__doc__ = _doc_from_base(base.AbstractAffine)


class TimeDecimation(base.AbstractTimeDecimation):
    def __init__(self, *args, **kwargs):
        super(TimeDecimation, self).__init__(*args, **kwargs)


TimeDecimation.__doc__ = _doc_from_base(base.AbstractTimeDecimation)


class Dense(AbstractRFIz, base.AbstractDense):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.synapse = synapse.Dense(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Dense.__doc__ = _doc_from_base(base.AbstractDense)


class Conv(AbstractRFIz, base.AbstractConv):
    def __init__(self, *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        self.synapse = synapse.Conv(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Conv.__doc__ = _doc_from_base(base.AbstractConv)


class ConvT(AbstractRFIz, base.AbstractConvT):
    def __init__(self, *args, **kwargs):
        super(ConvT, self).__init__(*args, **kwargs)
        self.synapse = synapse.ConvTranspose(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


ConvT.__doc__ = _doc_from_base(base.AbstractConvT)


class Pool(AbstractRFIz, base.AbstractPool):
    def __init__(self, *args, **kwargs):
        super(Pool, self).__init__(*args, **kwargs)
        self.synapse = synapse.Pool(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Pool.__doc__ = _doc_from_base(base.AbstractPool)


class Unpool(AbstractRFIz, base.AbstractUnpool):
    def __init__(self, *args, **kwargs):
        super(Unpool, self).__init__(*args, **kwargs)
        self.synapse = synapse.Unpool(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Unpool.__doc__ = _doc_from_base(base.AbstractUnpool)


class KWTA(AbstractRFIz, base.AbstractKWTA):
    def __init__(self, *args, **kwargs):
        super(KWTA, self).__init__(*args, **kwargs)
        self.synapse = synapse.Dense(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


KWTA.__doc__ = _doc_from_base(base.AbstractKWTA)


class Recurrent(AbstractRFIz, base.AbstractRecurrent):
    def __init__(self, *args, **kwargs):
        super(Recurrent, self).__init__(*args, **kwargs)
        self.input_synapse = synapse.Dense(**self.synapse_params)
        self.recurrent_synapse = synapse.Dense(**self.recurrent_params)
        self.input_synapse.pre_hook_fx = self.neuron.quantize_8bit
        self.recurrent_synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params
        del self.recurrent_params


Recurrent.__doc__ = _doc_from_base(base.AbstractRecurrent)
