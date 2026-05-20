# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""CUBA-Hoyer-LIF layer blocks"""

import torch

from . import base, cuba
from ..neuron import cuba_hoyer
from ..synapse import layer as synapse
from ..axon import Delay, delay


class AbstractCubaHoyer(torch.nn.Module):
    """Abstract block class for Current Based Leaky Integrator neuron. This
    should never be instantiated on it's own.
    """
    def __init__(self, *args, **kwargs):
        super(AbstractCubaHoyer, self).__init__(*args, **kwargs)
        if self.neuron_params is not None:
            self.neuron = cuba_hoyer.HoyerNeuron(**self.neuron_params)
        delay = kwargs['delay'] if 'delay' in kwargs.keys() else False
        self.delay = Delay(max_delay=62) if delay is True else None
        del self.neuron_params


def _doc_from_base(base_doc):
    """ """
    return base_doc.__doc__.replace(
        'Abstract', 'CUBA Hoyer LIF'
    ).replace(
        'neuron parameter', 'CUBA Hoyer LIF neuron parameter'
    ).replace(
        'This should never be instantiated on its own.',
        'The block is 8 bit quantization ready.'
    )


class Dense(AbstractCubaHoyer, base.AbstractDense):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.synapse = synapse.Dense(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params

Dense.__doc__ = _doc_from_base(base.AbstractDense)

class Conv(AbstractCubaHoyer, base.AbstractConv):
    def __init__(self, *args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        self.synapse = synapse.Conv(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params


Conv.__doc__ = _doc_from_base(base.AbstractConv)


def step_delay(module, x):
    """Step delay computation. This simulates the 1 timestep delay needed
    for communication between layers.

    Parameters
    ----------
    module: module
        python module instance
    x : torch.tensor
        Tensor data to be delayed.
    """
    if hasattr(module, 'delay_buffer') is False:
        module.delay_buffer = None
    persistent_state = hasattr(module, 'neuron') \
        and module.neuron.persistent_state is True
    if module.delay_buffer is not None:
        if module.delay_buffer.shape[0] != x.shape[0]:  # batch mismatch
            module.delay_buffer = None
    if persistent_state:
        delay_buffer = 0 if module.delay_buffer is None else module.delay_buffer
        module.delay_buffer = x[..., -1]
    x = delay(x, 1)
    if persistent_state:
        x[..., 0] = delay_buffer
    return x

class Pool(cuba.Pool):
    def __init__(self, *args, **kwargs):
        super(Pool, self).__init__(*args, **kwargs)
        self.hoyer_loss = 0.0

    def forward(self, x):
        """Forward computation method. The input must be in ``NCHWT`` format.
        """
        self.neuron.shape = x[0].shape
        z = self.synapse(x)
        # skip the neuron computation in the pooling layer
        # x = self.neuron(z)
        x = z
        if self.delay_shift is True:
            x = step_delay(self, x)
        if self.delay is not None:
            x = self.delay(x)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

Pool.__doc__ = _doc_from_base(base.AbstractPool)

class Affine(cuba.Affine):
    def __init__(self, *args, **kwargs):
        super(Affine, self).__init__(*args, **kwargs)

Affine.__doc__ = _doc_from_base(base.AbstractAffine)
