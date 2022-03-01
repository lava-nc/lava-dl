# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import unittest
import numpy as np
import torch

from lava.lib.dl.slayer.utils.time.shift import shift

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

seed = np.random.randint(1000)
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

input = torch.rand(10, 200, 1000)


class TestShift(unittest.TestCase):
    def test_shift_positive(self):
        shift_steps = 10
        output = shift(input, shift_steps)
        error = torch.abs(
            output[..., shift_steps:] - input[..., :-shift_steps]
        ).sum()
        self.assertTrue(
            error < 1e-6,
            f'Error in positive shift. Expected error<1e-6. Found {error=}.'
        )
        if torch.cuda.is_available():
            out_cuda = shift(input.to(device), shift_steps)
            error_cuda = torch.abs(
                out_cuda[..., shift_steps:].cpu()
                - input[..., :-shift_steps]
            ).sum()
            if verbose is True:
                print(f'{input=}')
                print(f'{out_cuda=}')
            self.assertTrue(
                error_cuda < 1e-6,
                f'Error in positive shift. Expected error_cuda<1e-6. '
                f'Found {error_cuda=}.'
            )

    def test_shift_negative(self):
        shift_steps = -10
        output = shift(input, shift_steps)
        error = torch.abs(
            output[..., :shift_steps] - input[..., -shift_steps:]
        ).sum()
        if verbose is True:
            print(f'{input=}')
            print(f'{output=}')
        self.assertTrue(
            error < 1e-6,
            f'Error in negative shift. Expected error<1e-6. Found {error=}.'
        )
        if torch.cuda.is_available():
            out_cuda = shift(input.to(device), shift_steps)
            error_cuda = torch.abs(
                out_cuda[..., :shift_steps].cpu()
                - input[..., -shift_steps:]
            ).sum()
            if verbose is True:
                print(f'{input=}')
                print(f'{out_cuda=}')
            self.assertTrue(
                error_cuda < 1e-6,
                f'Error in negative shift. Expected error_cuda<1e-6. '
                f'Found {error_cuda=}.'
            )

    def test_shift(self):
        shift_steps = torch.arange(input.shape[1]).to(input.dtype) \
            - input.shape[1] // 2
        output = shift(input, shift_steps).data.numpy()
        gt = np.zeros(input.shape)
        for i, s in enumerate(shift_steps.data.numpy().astype(int)):
            if s > 0:
                gt[:, i, s:] = input[:, i, :-s].data.numpy()
            elif s < 0:
                gt[:, i, :s] = input[:, i, -s:].data.numpy()
            else:
                gt[:, i] = input[:, i].data.numpy()

        if verbose is True:
            print(f'{input=}')
            print(f'{output=}')
            print(f'{gt=}')
        error = np.abs(output - gt).sum()
        self.assertTrue(
            error < 1e-6,
            f'Error in shift. Expected error<1e-6. Found {error=}.'
        )
        if torch.cuda.is_available():
            out_cuda = shift(
                input.to(device),
                shift_steps.to(device)
            ).cpu().data.numpy()
            if verbose is True:
                print(f'{out_cuda=}')
            error_cuda = np.abs(out_cuda - gt).sum()
            self.assertTrue(
                error_cuda < 1e-6,
                f'Error in shift. Expected error_cuda<1e-6. '
                f'Found {error_cuda=}.'
            )
