# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import lava.lib.dl.slayer as slayer

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

seed = np.random.randint(1000)
# seed = 902
np.random.seed(seed)
if verbose:
    print(f'{seed=}')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    if verbose:
        print(
            'CUDA is not available in the system. '
            'Testing for CPU version only.'
        )
    device = torch.device('cpu')

# neuron parameters
threshold = 1
scale = (1 << 12)
decay = np.random.random() * scale
decay = torch.FloatTensor([decay]).to(device)
state = torch.FloatTensor([0]).to(device)

# create input
time = torch.FloatTensor(np.arange(200)).to(device)
# expand to (batch, neuron, time) tensor
spike_input = torch.autograd.Variable(
    torch.zeros([5, 4, len(time)]),
    requires_grad=True
).to(device)
spike_input.data[..., np.random.randint(spike_input.shape[-1], size=5)] = 1
weight = torch.FloatTensor(
    5 * np.random.random(size=spike_input.shape[-1]) - 0.5
).reshape([1, 1, spike_input.shape[-1]]).to(device)
w_input = slayer.utils.quantize(weight) * spike_input


if verbose:
    print(f'{decay=}')

# get the dynamics response
output0 = slayer.neuron.dynamics.leaky_integrator.dynamics(
    w_input, decay=decay, state=state, w_scale=scale,
)
output = slayer.neuron.dynamics.leaky_integrator.dynamics(
    w_input, decay=decay, state=state, w_scale=scale, threshold=threshold,
    debug=True
)
spike = (output >= threshold).to(output.dtype)


class TestIF(unittest.TestCase):
    def test_input_output_range(self):
        if verbose:
            print(spike_input.sum(), spike_input.flatten())
        if verbose:
            print(spike.sum(), spike.flatten())

        self.assertTrue(
            spike_input.sum().item() > 0,
            'There was zero input spike. Check the test setting.'
        )
        self.assertTrue(
            spike.sum().item() > 0,
            'There was zero ouptut spike. Check the test setting.'
        )

    def test_leak(self):
        leak_num = output0[..., 1:] - w_input[..., 1:]
        leak_den = output0[..., :-1]
        valid = torch.abs(leak_den) > 10 / scale
        est_decay = torch.mean(1 - leak_num[valid] / leak_den[valid]) * scale
        rel_error = np.abs(
            (est_decay.item() - decay.item()) / max(decay.item(), 512)
        )
        if verbose:
            print(f'{rel_error=}')
            print(f'{est_decay=}')
            print(f'{decay=}')

        self.assertTrue(
            rel_error < 1e-1,  # the estimate is crude
            f'Expected estimated decay to match. '
            f'Found {est_decay=} and {decay=}'
        )

    def test_reset(self):
        spike_inds = (w_input[..., 1:] == output[..., 1:])
        spike_template = spike[..., :-1]
        spike_template[spike_inds] = 0
        error = torch.norm(spike_template).item()

        if verbose:
            print(f'{error=}')

        self.assertTrue(
            error < 1e-3,
            f'Expect reset points to match. Found {error=}.'
        )

    def test_integer_states(self):
        # there should be no quantization error
        # when states are scaled with s_scale
        output_error = torch.norm(torch.floor(output * scale) - output * scale)

        self.assertTrue(
            output_error < 1e-5,
            f'Voltage calculation has issues with scaling. '
            f'De-Scaling must result in integer states. '
            f'Error was {output_error}'
        )

    def test_backward(self):
        spike_target = spike.clone().detach()
        output_target = output.clone().detach()

        spike_target[
            ..., np.random.randint(spike_input.shape[-1], size=5)
        ] = 1
        output_target[
            ..., np.random.randint(spike_input.shape[-1], size=5)
        ] -= 1

        loss = F.mse_loss(spike, spike_target) \
            + F.mse_loss(output, output_target)
        loss.backward()

        # just looking for errors
        # self.assertTrue(True, 'Encountered errors.')
