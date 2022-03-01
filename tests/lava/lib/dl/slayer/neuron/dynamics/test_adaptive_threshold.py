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
decay = int(np.random.random() * scale)
decay = torch.FloatTensor([decay]).to(device)
th_decay = int(np.random.random() * scale)
th_decay = torch.FloatTensor([th_decay]).to(device)
ref_decay = int(np.random.random() * scale)
ref_decay = torch.FloatTensor([ref_decay]).to(device)
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

# get the dynamics response
voltage = slayer.neuron.dynamics.leaky_integrator.dynamics(
    w_input, decay=decay, state=state, w_scale=scale,
)
th, ref = slayer.neuron.dynamics.adaptive_threshold.dynamics(
    voltage,                      # dynamics state
    ref_state=state,              # previous refractory state
    ref_decay=ref_decay,          # refractory decay
    th_state=state + threshold,   # previous threshold state
    th_decay=th_decay,            # threshold decay
    th_scale=0.5 * threshold,     # threshold step
    th0=threshold,                # threshold stable state
    w_scale=scale,                # fixed precision scaling
    debug=True
)
spike = (voltage >= (th + ref)).to(voltage.dtype)


class TestAdTh(unittest.TestCase):
    def test_input_voltage_range(self):
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

    def test_th_leak(self):
        th_leak_num = th[..., 1:] - threshold
        th_leak_den = th[..., :-1] - threshold
        th_valid = (torch.abs(th_leak_den) > 100 / scale) \
            & (spike[..., :-1] == 0)
        if torch.sum(th_valid) > 0:
            est_th_decay = torch.mean(
                1 - th_leak_num[th_valid] / th_leak_den[th_valid]
            ) * scale
            th_error = np.abs(
                (est_th_decay.item() - th_decay.item()) / th_decay.item()
            )
            if verbose:
                print(f'{th_error=}')
                print(f'{est_th_decay=}')
                print(f'{th_decay=}')

            self.assertTrue(
                th_error < 1e-1,  # the estimate is crude
                f'Expected estimated decay to match. '
                f'Found {est_th_decay=} and {th_decay=}'
            )

    def test_ref_leak(self):
        ref_leak_num = ref[..., 1:]
        ref_leak_den = ref[..., :-1]
        ref_valid = (torch.abs(ref_leak_den) > 100 / scale) \
            & (spike[..., :-1] == 0)
        if torch.sum(ref_valid) > 0:
            est_ref_decay = torch.mean(
                1 - ref_leak_num[ref_valid] / ref_leak_den[ref_valid]
            ) * scale
            ref_error = np.abs(
                (est_ref_decay.item() - ref_decay.item())
                / max(ref_decay.item(), 512)
            )
            if verbose:
                print(f'{ref_error=}')
                print(f'{est_ref_decay=}')
                print(f'{ref_decay=}')

            self.assertTrue(
                ref_error < 1e-1,  # the estimate is crude
                f'Expected estimated decay to match. '
                f'Found {est_ref_decay=} and {ref_decay=}'
            )

    def test_integer_states(self):
        # there should be no quantization error
        # when states are scaled with s_scale
        voltage_error = torch.norm(
            torch.floor(voltage * scale) - voltage * scale
        )

        self.assertTrue(
            voltage_error < 1e-5,
            f'Voltage calculation has issues with scaling. '
            f'De-Scaling must result in integer states. '
            f'Error was {voltage_error}'
        )

    def test_backward(self):
        spike_target = spike.clone().detach()
        voltage_target = voltage.clone().detach()

        spike_target[
            ..., np.random.randint(spike_input.shape[-1], size=5)
        ] = 1
        voltage_target[
            ..., np.random.randint(spike_input.shape[-1], size=5)
        ] -= 1

        loss = F.mse_loss(spike, spike_target) \
            + F.mse_loss(voltage, voltage_target)
        loss.backward()

        # just looking for errors
        # self.assertTrue(True, 'Encountered errors.')
