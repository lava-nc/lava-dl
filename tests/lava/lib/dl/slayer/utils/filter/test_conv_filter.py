# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Test for conv and corr filters."""

import sys
import unittest

import numpy as np
import torch
from lava.lib.dl.slayer.utils.filter import conv, corr

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

input = torch.rand(200, 1000)
filter = torch.rand(50)

conv_gt = np.zeros(input.shape)
for i in range(input.shape[0]):
    conv_gt[i] = np.convolve(input[i].cpu().data.numpy(),
                             filter.cpu().data.numpy(),
                             'full')[:1000]

corr_gt = np.zeros(input.shape)
for i in range(input.shape[0]):
    corr_gt[i] = np.convolve(input[i].cpu().data.numpy(),
                             filter.cpu().data.numpy()[::-1],
                             'full')[-1000:]


class TestConv(unittest.TestCase):
    def test_conv(self):
        out = conv(input, filter)
        error = np.abs(conv_gt - out.data.numpy()).mean()
        if verbose is True:
            print(f'Conv calculation error: {error}')
        self.assertTrue(
            error < 1e-4,
            f'CPU conv calculation does not match with numpy.\n{error=}'
        )
        if torch.cuda.is_available():
            out_cuda = conv(input.to(device), filter.to(device))
            error_cuda = np.abs(conv_gt - out_cuda.cpu().data.numpy()).mean()
            if verbose is True:
                print(f'Conv(GPU) calculation error: {error_cuda}')
            self.assertTrue(
                error_cuda < 1e-4,
                f'GPU conv calculation does not match with numpy.\n'
                f'{error_cuda=}'
            )


class TestCorr(unittest.TestCase):
    def test_corr(self):
        out = corr(input, filter)
        error = np.abs(corr_gt - out.data.numpy()).mean()
        if verbose is True:
            print(f'Corr calculation error: {error}')
        self.assertTrue(
            error < 1e-4,
            f'CPU corr calculation does not match with numpy.\n{error=}'
        )
        if torch.cuda.is_available():
            out_cuda = corr(input.to(device), filter.to(device))
            error_cuda = np.abs(corr_gt - out_cuda.cpu().data.numpy()).mean()
            if verbose is True:
                print(f'Corr(GPU) calculation error: {error_cuda}')
            self.assertTrue(
                error_cuda < 1e-4,
                f'GPU corr calculation does not match with numpy.'
                f'\n{error_cuda=}'
            )
