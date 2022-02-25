# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lava.lib.dl.slayer.neuron import rf

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

seed = np.random.randint(1000)
# seed = 133
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
decay = np.random.random() * 0.1
period = np.random.randint(4, 50)

# create input
time = torch.FloatTensor(np.arange(200)).to(device)
# expand to (batch, neuron, time) tensor
spike_input = torch.autograd.Variable(
    torch.zeros([5, 4, len(time)]), requires_grad=True
).to(device)
spike_input.data[..., np.random.randint(spike_input.shape[-1], size=5)] = 1

real_weight = torch.FloatTensor(
    5 * np.random.random(size=spike_input.shape[-1]) - 0.5
).reshape(
    [1, 1, spike_input.shape[-1]]
).to(device)

imag_weight = torch.FloatTensor(
    5 * np.random.random(size=spike_input.shape[-1]) - 0.5
).reshape(
    [1, 1, spike_input.shape[-1]]
).to(device)

# initialize neuron
neuron = rf.Neuron(threshold, period, decay, persistent_state=True).to(device)
quantized_real_weight = neuron.quantize_8bit(real_weight)
quantized_imag_weight = neuron.quantize_8bit(imag_weight)
neuron.debug = True

real, imag = neuron.dynamics((
    quantized_real_weight * spike_input,
    quantized_imag_weight * spike_input
))
spike = neuron.spike(real, imag)


class TestRF(unittest.TestCase):
    def test_input_range(self):
        # not doing it for output spikes because RF neurons
        # tend to spike sparsely.
        if verbose:
            print(spike.sum(), spike.flatten())

        self.assertTrue(
            spike_input.sum().item() > 0,
            'There was zero input spike. Check the test setting.'
        )

    def test_properties(self):
        _ = neuron.lam
        _ = neuron.decay
        _ = neuron.period
        _ = neuron.frequency
        _ = neuron.cx_sin_decay
        _ = neuron.cx_cos_decay
        _ = neuron.scale
        _ = neuron.shape
        _ = neuron.device

        # just looking for errors
        self.assertTrue(True, 'Encountered errors.')

    def test_batch_consistency(self):
        spike_var = torch.norm(torch.var(spike, dim=0)).item()
        real_var = torch.norm(torch.var(real, dim=0)).item()
        imag_var = torch.norm(torch.var(imag, dim=0)).item()
        self.assertTrue(
            spike_var < 1e-5,
            f'Spike variation across batch dimension is inconsistent. '
            f'Variance was {spike_var}. Expected 0.'
        )
        self.assertTrue(
            real_var < 1e-5,
            f'Real state variation across batch dimension is inconsistent. '
            f'Variance was {real_var}. Expected 0.'
        )
        self.assertTrue(
            imag_var < 1e-5,
            f'Voltage variation across batch dimension is inconsistent. '
            f'Variance was {imag_var}. Expected 0.'
        )

    def test_integer_states(self):
        # there should be no quantization error when
        # states are scaled with s_scale
        real_error = torch.norm(
            torch.floor(real * neuron.s_scale) - real * neuron.s_scale
        )
        imag_error = torch.norm(
            torch.floor(imag * neuron.s_scale) - imag * neuron.s_scale
        )

        self.assertTrue(
            real_error < 1e-5,
            f'Real calculation has issues with scaling. '
            f'De-Scaling must result in integer states. '
            f'Error was {real_error}'
        )
        self.assertTrue(
            imag_error < 1e-5,
            f'Imag calculation has issues with scaling. '
            f'De-Scaling must result in integer states. '
            f'Error was {imag_error}'
        )

    def test_persistent_state(self):
        # clear previous persistent state
        neuron.real_state *= 0
        neuron.imag_state *= 0

        # break the calculation into two parts: before ind and after ind
        ind = int(np.random.random() * (spike_input.shape[-1] - 1)) + 1
        # ind = 57
        real0, imag0 = neuron.dynamics((
            quantized_real_weight[..., :ind] * spike_input[..., :ind],
            quantized_imag_weight[..., :ind] * spike_input[..., :ind]
        ))
        spike0 = neuron.spike(real0, imag0)
        real1, imag1 = neuron.dynamics((
            quantized_real_weight[..., ind:] * spike_input[..., ind:],
            quantized_imag_weight[..., ind:] * spike_input[..., ind:]
        ))
        spike1 = neuron.spike(real1, imag1)

        spike_error = (
            torch.norm(spike[..., :ind] - spike0)
            + torch.norm(spike[..., ind:] - spike1)
        ).item()
        real_error = (
            torch.norm(real[..., :ind] - real0)
            + torch.norm(real[..., ind:] - real1)
        ).item()
        imag_error = (
            torch.norm(imag[..., :ind] - imag0)
            + torch.norm(imag[..., ind:] - imag1)
        ).item()

        if verbose:
            print(ind)
            if spike_error >= 1e-5:
                print('Persistent spike states')
                print(
                    spike[0, 0, ind - 10:ind + 10].cpu().data.numpy().tolist()
                )
                print(spike0[0, 0, -10:].cpu().data.numpy().tolist())
                print(spike1[0, 0, :10].cpu().data.numpy().tolist())
            if real_error >= 1e-5:
                print('Persistent real states')
                print((
                    neuron.s_scale * real[0, 0, ind - 10:ind + 10]
                ).cpu().data.numpy().astype(int).tolist())
                print((
                    neuron.s_scale * real0[0, 0, -10:]
                ).cpu().data.numpy().astype(int).tolist())
                print((
                    neuron.s_scale * real1[0, 0, :10]
                ).cpu().data.numpy().astype(int).tolist())
            if imag_error >= 1e-5:
                print('Persistent imag states')
                print((
                    neuron.s_scale * imag[0, 0, ind - 10:ind + 10]
                ).cpu().data.numpy().astype(int).tolist())
                print((
                    neuron.s_scale * imag0[0, 0, -10:]
                ).cpu().data.numpy().astype(int).tolist())
                print((
                    neuron.s_scale * imag1[0, 0, :10]
                ).cpu().data.numpy().astype(int).tolist())

        if verbose:
            if bool(os.environ.get('DISPLAY', None)):
                plt.figure()
                plt.plot(
                    time.cpu().data.numpy(),
                    imag[0, 0].cpu().data.numpy(),
                    label='imag'
                )
                plt.plot(
                    time[:ind].cpu().data.numpy(),
                    imag0[0, 0].cpu().data.numpy(),
                    label=':ind'
                )
                plt.plot(
                    time[ind:].cpu().data.numpy(),
                    imag1[0, 0].cpu().data.numpy(),
                    label='ind:'
                )
                plt.xlabel('time')
                plt.legend()

                plt.figure()
                plt.plot(
                    time.cpu().data.numpy(),
                    real[0, 0].cpu().data.numpy(),
                    label='real'
                )
                plt.plot(
                    time[:ind].cpu().data.numpy(),
                    real0[0, 0].cpu().data.numpy(),
                    label=':ind'
                )
                plt.plot(
                    time[ind:].cpu().data.numpy(),
                    real1[0, 0].cpu().data.numpy(),
                    label='ind:'
                )

                plt.plot(
                    time[spike[0, 0] > 0].cpu().data.numpy(),
                    0 * spike[0, 0][spike[0, 0] > 0].cpu().data.numpy(),
                    '.', markersize=12, label='spike'
                )
                plt.plot(
                    time[:ind][spike0[0, 0] > 0].cpu().data.numpy(),
                    0 * spike0[0, 0][spike0[0, 0] > 0].cpu().data.numpy(),
                    '.', label=':ind'
                )
                plt.plot(
                    time[ind:][spike1[0, 0] > 0].cpu().data.numpy(),
                    0 * spike1[0, 0][spike1[0, 0] > 0].cpu().data.numpy(),
                    '.', label='ind:'
                )
                plt.xlabel('time')
                plt.legend()
                plt.show()

        self.assertTrue(
            spike_error < 1e-5,
            f'Persistent state has errors in spike calculation. '
            f'Error was {spike_error}.'
            f'{seed=}'
        )
        self.assertTrue(
            real_error < 1e-5,
            f'Persistent state has errors in real calculation. '
            f'Error was {real_error}.'
            f'{seed=}'
        )
        self.assertTrue(
            imag_error < 1e-5,
            f'Persistent state has errors in imag calculation. '
            f'Error was {imag_error}.'
            f'{seed=}'
        )

    def test_backward(self):
        spike_target = spike.clone().detach()
        real_target = real.clone().detach()
        imag_target = imag.clone().detach()

        spike_target[
            ..., np.random.randint(spike_input.shape[-1], size=5)
        ] = 1
        real_target[
            ..., np.random.randint(spike_input.shape[-1], size=5)
        ] -= 1
        imag_target[
            ..., np.random.randint(spike_input.shape[-1], size=5)
        ] -= -1

        loss = F.mse_loss(spike, spike_target) \
            + F.mse_loss(real, real_target) \
            + F.mse_loss(imag, imag_target)
        loss.backward()

        # just looking for errors
        self.assertTrue(True, 'Encountered errors.')

    def test_graded_spikes(self):
        # TODO: after further study of network behavior with graded spikes.
        pass
