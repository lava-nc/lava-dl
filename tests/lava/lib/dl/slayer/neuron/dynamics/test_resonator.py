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
decay = torch.FloatTensor([np.random.random() * scale]).to(device)
phi = 2 * np.pi / (4 + 96 * np.random.random())
sin_decay = slayer.utils.quantize((scale - decay) * np.sin(phi)).to(device)
cos_decay = slayer.utils.quantize((scale - decay) * np.cos(phi)).to(device)
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
    5 * np.random.random(size=re_input.shape[-1]) - 0.5
).reshape([1, 1, re_input.shape[-1]]).to(device)
im_weight = torch.FloatTensor(
    5 * np.random.random(size=im_input.shape[-1]) - 0.5
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
    debug=True
)

# get the dynamics response
re_iz, im_iz = slayer.neuron.dynamics.resonator.dynamics(
    re_input, im_input,
    sin_decay, cos_decay,
    real_state=state,
    imag_state=state,
    w_scale=scale,
    threshold=threshold,
    debug=True
)

spike = (im_iz >= threshold).to(re.dtype)


class TestRF(unittest.TestCase):
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

    def test_reset(self):
        # TODO
        pass

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
