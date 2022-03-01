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
threshold = 0.1
scale = (1 << 12)
rand_param = np.random.random()
decay = torch.FloatTensor([0.1 * rand_param * scale]).to(device)
phi = 2 * np.pi / (4 + 46 * rand_param)
sin_decay = slayer.utils.quantize((scale - decay) * np.sin(phi)).to(device)
cos_decay = slayer.utils.quantize((scale - decay) * np.cos(phi)).to(device)
th_decay = int(np.random.random() * scale)
th_decay = torch.FloatTensor([th_decay]).to(device)
ref_decay = int(np.random.random() * scale)
ref_decay = torch.FloatTensor([ref_decay]).to(device)
state = torch.FloatTensor([0]).to(device)

# create input
time = torch.FloatTensor(np.arange(200)).to(device)
# expand to (batch, neuron, time) tensor
re_input = torch.autograd.Variable(
    torch.zeros([5, 4, len(time)]),
    requires_grad=True
).to(device)
re_input.data[..., np.random.randint(re_input.shape[-1], size=5)] = 1
im_input = torch.autograd.Variable(
    torch.zeros([5, 4, len(time)]),
    requires_grad=True
).to(device)
im_input.data[..., np.random.randint(im_input.shape[-1], size=5)] = 1
re_weight = torch.FloatTensor(
    10 * np.random.random(size=re_input.shape[-1]) - 0.5
).reshape([1, 1, re_input.shape[-1]]).to(device)
im_weight = torch.FloatTensor(
    10 * np.random.random(size=im_input.shape[-1]) - 0.5
).reshape([1, 1, im_input.shape[-1]]).to(device)
re_w_input = slayer.utils.quantize(re_weight) * re_input
im_w_input = slayer.utils.quantize(im_weight) * im_input

# get the dynamics response
re, im = slayer.neuron.dynamics.resonator.dynamics(
    re_input, im_input,
    sin_decay, cos_decay,
    real_state=state,
    imag_state=state,
    w_scale=scale,
)
th, ref = slayer.neuron.dynamics.adaptive_phase_th.dynamics(
    re, im,
    im_state=state,
    ref_state=state, ref_decay=ref_decay,
    th_state=state + threshold, th_decay=th_decay,
    th_scale=5 * threshold,
    th0=threshold,
    w_scale=scale,
    debug=True
)
spike = slayer.spike.complex.Spike.apply(
    re, im, th + ref,
    1,  # tau_rho: gradient relaxation constant
    1,  # scale_rho: gradient scale constant
    False,  # graded_spike: graded or binary spike
    0,  # voltage_last: voltage at t=-1
    1,  # scale: graded spike scale
)


class TestAdRF(unittest.TestCase):
    def test_input_output_range(self):
        if verbose:
            print(re_input.sum(), re_input.flatten())
        if verbose:
            print(im_input.sum(), im_input.flatten())

        self.assertTrue(
            re_input.sum().item() > 0,
            'There was zero real input spike. Check the test setting.'
        )
        self.assertTrue(
            im_input.sum().item() > 0,
            'There was zero imaginary input spike. Check the test setting.'
        )

    def test_leak(self):
        real = re - re_input
        imag = im - im_input
        mag = torch.sqrt(real * real + imag * imag)
        phase = torch.atan2(real, imag)
        leak_num = mag[..., 1:]
        leak_den = mag[..., :-1]
        valid = (torch.abs(leak_den) > 100 / scale) \
            & (re_input[..., :-1] == 0) & (im_input[..., :-1] == 0)
        if torch.sum(valid) > 0:
            est_decay = torch.mean(
                1 - leak_num[valid] / leak_den[valid]
            ) * scale
            est_phase = torch.mean(
                (phase[..., :-1] - phase[..., 1:])[valid] % (2 * np.pi)
            )
            mag_error = np.abs(
                (est_decay.item() - decay.item()) / max(decay.item(), 512)
            )
            phase_error = np.abs(
                (est_phase.item() - phi) / phi
            )
            if verbose:
                print(f'{mag_error=}')
                print(f'{est_decay=}')
                print(f'{decay=}')
                print(f'{phase_error=}')
                print(f'{est_phase=}')
                print(f'{phi=}')

            self.assertTrue(
                mag_error < 1e-1,  # the estimate is crude
                f'Expected estimated decay to match. '
                f'Found {est_decay=} and {decay=}'
            )
            self.assertTrue(
                phase_error < 1,  # the estimate is crude
                f'Expected estimated phase to match. '
                f'Found {est_phase=} and {phi=}'
            )

    def test_integer_states(self):
        # there should be no quantization error
        # when states are scaled with s_scale
        re_error = torch.norm(torch.floor(re * scale) - re * scale)
        im_error = torch.norm(torch.floor(im * scale) - im * scale)

        self.assertTrue(
            re_error < 1e-5,
            f'Real calculation has issues with scaling. '
            f'De-Scaling must result in integer states. '
            f'Error was {re_error}'
        )
        self.assertTrue(
            im_error < 1e-5,
            f'Imag calculation has issues with scaling. '
            f'De-Scaling must result in integer states. '
            f'Error was {im_error}'
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

    def test_backward(self):
        re_target = re.clone().detach()
        im_target = im.clone().detach()

        re_target[
            ..., np.random.randint(re_input.shape[-1], size=5)
        ] -= 1
        im_target[
            ..., np.random.randint(im_input.shape[-1], size=5)
        ] = 1

        loss = F.mse_loss(re, re_target) + F.mse_loss(im, im_target)
        loss.backward()

        # just looking for errors
        # self.assertTrue(True, 'Encountered errors.')
