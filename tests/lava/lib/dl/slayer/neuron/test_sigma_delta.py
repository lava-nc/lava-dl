# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lava.lib.dl.slayer.neuron import rf, sigma_delta

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
decay_min = np.random.random() * 0.01
decay_max = np.random.random() * 0.1
period_min = np.random.randint(4, 50)
period_max = np.random.randint(50, 150)

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
neuron = rf.Neuron(
    threshold,
    [period_min, period_max],
    [decay_min, decay_max],
    persistent_state=True,
    shared_param=False,
).to(device)

sd_neuron = sigma_delta.Neuron(
    threshold=threshold,
    activation=lambda x: x,
    persistent_state=True
).to(device)
sd_neuron.debug = True

quantized_real_weight = neuron.quantize_8bit(real_weight)
quantized_imag_weight = neuron.quantize_8bit(imag_weight)

re_spike = torch.autograd.Variable(
    quantized_real_weight * spike_input,
    requires_grad=True,
)
im_spike = torch.autograd.Variable(
    quantized_imag_weight * spike_input,
    requires_grad=True,
)

# get the neuron response for full input
real, _ = neuron.dynamics((re_spike, im_spike))
real_delta = sd_neuron.delta(real)
real_sigma = sd_neuron.sigma(real_delta)

real.retain_grad()
real_sigma.retain_grad()

real_target = real_sigma.clone().detach()

target_perturbation = 5 * np.random.random() - 0.5
real_target[:, 0, -5] -= target_perturbation
real_target[:, 1, -7] -= target_perturbation
real_target[:, 2, -4] -= target_perturbation
real_target[:, 3, -20] -= target_perturbation

loss = F.mse_loss(real_sigma, real_target)
loss.backward()


class TestCUBA(unittest.TestCase):
    def test_properties(self):
        _ = neuron.weight_exponent
        _ = neuron.v_th_mant
        _ = neuron.scale
        _ = neuron.shape
        _ = neuron.device

        # just looking for errors
        self.assertTrue(True, 'Encountered errors.')

    def test_batch_consistency(self):
        real_delta_var = torch.norm(torch.var(real_delta, dim=0)).item()
        real_sigma_var = torch.norm(torch.var(real_sigma, dim=0)).item()
        self.assertTrue(
            real_delta_var < 1e-5,
            f'real_delta variation across batch dimension is inconsistent. '
            f'Variance was {real_delta_var}. Expected 0.'
        )
        self.assertTrue(
            real_sigma_var < 1e-5,
            f'real_sigma variation across batch dimension is inconsistent. '
            f'Variance was {real_sigma_var}. Expected 0.'
        )

    def test_integer_states(self):
        # TODO
        pass

    def test_persistent_state(self):
        # clear previous persistent state
        neuron.real_state *= 0
        neuron.imag_state *= 0
        sd_neuron.delta.pre_state *= 0
        sd_neuron.delta.residual_state *= 0
        sd_neuron.delta.error_state *= 0
        sd_neuron.sigma.pre_state *= 0

        # break the calculation into two parts: before ind and after ind
        ind = int(np.random.random() * (spike_input.shape[-1] - 1)) + 1
        real_sigma0 = sd_neuron.sigma(sd_neuron.delta(real[..., :ind]))
        real_sigma1 = sd_neuron.sigma(sd_neuron.delta(real[..., ind:]))

        real_sigma_error = (
            torch.norm(real_sigma[..., :ind] - real_sigma0)
            + torch.norm(real_sigma[..., ind:] - real_sigma1)
        ).item()

        if verbose:
            print(ind)
            if real_sigma_error >= 1e-5:
                print('Persistent real_sigma states')
                print(
                    real_sigma[
                        0, 0, ind - 10: ind + 10
                    ].cpu().data.numpy().tolist()
                )
                print(real_sigma0[0, 0, -10:].cpu().data.numpy().tolist())
                print(real_sigma1[0, 0, :10].cpu().data.numpy().tolist())

        if verbose:
            if bool(os.environ.get('DISPLAY', None)):
                plt.figure()
                plt.plot(
                    time.cpu().data.numpy(),
                    real_sigma[0, 0].cpu().data.numpy(),
                    label='real_sigma'
                )
                plt.plot(
                    time[:ind].cpu().data.numpy(),
                    real_sigma0[0, 0].cpu().data.numpy(),
                    label=':ind'
                )
                plt.plot(
                    time[ind:].cpu().data.numpy(),
                    real_sigma1[0, 0].cpu().data.numpy(),
                    label='ind:'
                )
                plt.xlabel('time')
                plt.legend()
                plt.show()

        self.assertTrue(
            real_sigma_error < 1e-5,
            f'Persistent state has errors in sigma delta calculation. '
            f'Error was {real_sigma_error}'
        )

    def test_backward(self):
        error = torch.norm(real.grad - real_sigma.grad).item()

        self.assertTrue(error < 1e-6, 'Grads do not match')
