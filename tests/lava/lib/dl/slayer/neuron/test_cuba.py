# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lava.lib.dl.slayer.neuron import cuba

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
current_decay = np.random.random()
voltage_decay = np.random.random()

# create input
time = torch.FloatTensor(np.arange(200)).to(device)
# expand to (batch, neuron, time) tensor
spike_input = torch.autograd.Variable(
    torch.zeros([5, 4, len(time)]), requires_grad=True
).to(device)
spike_input.data[..., np.random.randint(spike_input.shape[-1], size=5)] = 1
weight = torch.FloatTensor(
    5 * np.random.random(size=spike_input.shape[-1]) - 0.5
).reshape(
    [1, 1, spike_input.shape[-1]]
).to(device)


# initialize neuron
neuron = cuba.Neuron(
    threshold, current_decay, voltage_decay, persistent_state=True
).to(device)
quantized_weight = neuron.quantize_8bit(weight)
neuron.debug = True

# get the neuron response for full input
voltage, current = neuron.dynamics(quantized_weight * spike_input)
spike = neuron.spike(voltage)


class TestCUBA(unittest.TestCase):
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

    def test_properties(self):
        _ = neuron.weight_exponent
        _ = neuron.v_th_mant
        _ = neuron.cx_current_decay
        _ = neuron.cx_voltage_decay
        _ = neuron.scale
        _ = neuron.shape
        _ = neuron.device

        # just looking for errors
        self.assertTrue(True, 'Encountered errors.')

    def test_batch_consistency(self):
        spike_var = torch.norm(torch.var(spike, dim=0)).item()
        voltage_var = torch.norm(torch.var(voltage, dim=0)).item()
        current_var = torch.norm(torch.var(current, dim=0)).item()
        self.assertTrue(
            spike_var < 1e-5,
            f'Spike variation across batch dimension is inconsistent. '
            f'Variance was {spike_var}. Expected 0.'
        )
        self.assertTrue(
            current_var < 1e-5,
            f'Current variation across batch dimension is inconsistent. '
            f'Variance was {current_var}. Expected 0.'
        )
        self.assertTrue(
            voltage_var < 1e-5,
            f'Voltage variation across batch dimension is inconsistent. '
            f'Variance was {voltage_var}. Expected 0.'
        )

    def test_integer_states(self):
        # there should be no quantization error when
        # states are scaled with s_scale
        voltage_error = torch.norm(
            torch.floor(voltage * neuron.s_scale)
            - voltage * neuron.s_scale
        )
        current_error = torch.norm(
            torch.floor(current * neuron.s_scale)
            - current * neuron.s_scale
        )

        self.assertTrue(
            voltage_error < 1e-5,
            f'Voltage calculation has issues with scaling. '
            f'De-Scaling must result in integer states. '
            f'Error was {voltage_error}'
        )
        self.assertTrue(
            current_error < 1e-5,
            f'Vurrent calculation has issues with scaling. '
            f'De-Scaling must result in integer states. '
            f'Error was {current_error}'
        )

    def test_persistent_state(self):
        # clear previous persistent state
        neuron.current_state *= 0
        neuron.voltage_state *= 0

        # break the calculation into two parts: before ind and after ind
        ind = int(np.random.random() * (spike_input.shape[-1] - 1)) + 1
        voltage0, current0 = neuron.dynamics(
            quantized_weight[..., :ind] * spike_input[..., :ind]
        )
        spike0 = neuron.spike(voltage0)
        voltage1, current1 = neuron.dynamics(
            quantized_weight[..., ind:] * spike_input[..., ind:]
        )
        spike1 = neuron.spike(voltage1)

        spike_error = (
            torch.norm(spike[..., :ind] - spike0)
            + torch.norm(spike[..., ind:] - spike1)
        ).item()
        voltage_error = (
            torch.norm(voltage[..., :ind] - voltage0)
            + torch.norm(voltage[..., ind:] - voltage1)
        ).item()
        current_error = (
            torch.norm(current[..., :ind] - current0)
            + torch.norm(current[..., ind:] - current1)
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
            if voltage_error >= 1e-5:
                print('Persistent voltage states')
                print((
                    neuron.s_scale * voltage[0, 0, ind - 10:ind + 10]
                ).cpu().data.numpy().astype(int).tolist())
                print((
                    neuron.s_scale * voltage0[0, 0, -10:]
                ).cpu().data.numpy().astype(int).tolist())
                print((
                    neuron.s_scale * voltage1[0, 0, :10]
                ).cpu().data.numpy().astype(int).tolist())
            if current_error >= 1e-5:
                print('Persistent current states')
                print((
                    neuron.s_scale * current[0, 0, ind - 10:ind + 10]
                ).cpu().data.numpy().astype(int).tolist())
                print((
                    neuron.s_scale * current0[0, 0, -10:]
                ).cpu().data.numpy().astype(int).tolist())
                print((
                    neuron.s_scale * current1[0, 0, :10]
                ).cpu().data.numpy().astype(int).tolist())

        if verbose:
            if bool(os.environ.get('DISPLAY', None)):
                plt.figure()
                plt.plot(
                    time.cpu().data.numpy(),
                    current[0, 0].cpu().data.numpy(),
                    label='current'
                )
                plt.plot(
                    time[:ind].cpu().data.numpy(),
                    current0[0, 0].cpu().data.numpy(),
                    label=':ind'
                )
                plt.plot(
                    time[ind:].cpu().data.numpy(),
                    current1[0, 0].cpu().data.numpy(),
                    label='ind:'
                )
                plt.xlabel('time')
                plt.legend()

                plt.figure()
                plt.plot(
                    time.cpu().data.numpy(),
                    voltage[0, 0].cpu().data.numpy(),
                    label='voltage'
                )
                plt.plot(
                    time[:ind].cpu().data.numpy(),
                    voltage0[0, 0].cpu().data.numpy(),
                    label=':ind'
                )
                plt.plot(
                    time[ind:].cpu().data.numpy(),
                    voltage1[0, 0].cpu().data.numpy(),
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
            voltage_error < 1e-5,
            f'Persistent state has errors in voltage calculation. '
            f'Error was {voltage_error}.'
            f'{seed=}'
        )
        self.assertTrue(
            current_error < 1e-5,
            f'Persistent state has errors in current calculation. '
            f'Error was {current_error}.'
            f'{seed=}'
        )

    def test_backward(self):
        spike_target = spike.clone().detach()
        current_target = current.clone().detach()
        voltage_target = voltage.clone().detach()

        spike_target[
            ...,
            np.random.randint(spike_input.shape[-1], size=5)
        ] = 1
        current_target[
            ...,
            np.random.randint(spike_input.shape[-1], size=5)
        ] -= 1
        voltage_target[
            ...,
            np.random.randint(spike_input.shape[-1], size=5)
        ] -= -1

        loss = F.mse_loss(spike, spike_target) \
            + F.mse_loss(current, current_target) \
            + F.mse_loss(voltage, voltage_target)
        loss.backward()

        # just looking for errors
        self.assertTrue(True, 'Encountered errors.')

    def test_graded_spikes(self):
        # TODO: after further study of network behavior with graded spikes.
        pass
