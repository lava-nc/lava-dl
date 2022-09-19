# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import unittest
import torch

from src.lava.lib.dl.slayer.utils.recurrent import (
    custom_recurrent,
    custom_recurrent_ground_truth_1,
    custom_recurrent_ground_truth_2,
)
from src.lava.lib.dl.slayer.neuron.cuba import Neuron
from src.lava.lib.dl.slayer.synapse.layer import Dense

verbose = True if (("-v" in sys.argv) or ("--verbose" in sys.argv)) else False


class TestRecurrent(unittest.TestCase):
    def test_recurrent(self):

        if verbose is True:
            print("testing recurrent")

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            if verbose:
                print(
                    'CUDA is not available in the system. '
                    'Testing for CPU version only.'
                )
            device = torch.device('cpu')

        torch.manual_seed(298101)

        batch_size = 2
        n_time_steps = 128
        n_neur = 10

        cuba_params = {
            "threshold": 1.25,
            "current_decay": 0.25,
            "voltage_decay": 0.25,
            "shared_param": True,
            "requires_grad": True,
            "graded_spike": False,
        }

        z = torch.randn(
            (
                batch_size,
                n_neur,
                n_time_steps,
            )
        ).to(device)
        if verbose is True:
            print(f"{z.shape=}")

        # Create three independent neuron and recurrent_syanpse
        # - one for each version of custom_recurrent
        neuron_list = list(
            map(
                lambda _: Neuron(**cuba_params, persistent_state=True).to(
                    device
                ),
                range(3),
            )
        )
        recurrent_synapse_list = list(
            map(
                lambda _: Dense(in_neurons=n_neur, out_neurons=n_neur).to(
                    device
                ),
                range(3),
            )
        )
        recurrent_synapse_list[1].weight.data = (
            recurrent_synapse_list[0].weight.data.detach().clone()
        )
        recurrent_synapse_list[2].weight.data = (
            recurrent_synapse_list[0].weight.data.detach().clone()
        )

        for recurrent_synapse, neuron in zip(
            recurrent_synapse_list, neuron_list
        ):
            recurrent_synapse.pre_hook_fx = neuron.quantize_8bit

        if verbose is True:
            list(
                map(
                    lambda i: print(
                        f"recurrent_synapse_list[{i}].pre_hook_fx"
                        + "={recurrent_synapse_list[i].pre_hook_fx}"
                    ),
                    range(3),
                )
            )

        custom_recurrent_output = custom_recurrent(
            z, neuron_list[0], recurrent_synapse_list[0]
        )

        custom_recurrent_ground_truth_1_output = (
            custom_recurrent_ground_truth_1(
                z, neuron_list[1], recurrent_synapse_list[1]
            )
        )

        custom_recurrent_ground_truth_2_output = (
            custom_recurrent_ground_truth_2(
                z, neuron_list[2], recurrent_synapse_list[2]
            )
        )

        forward_error_1 = torch.norm(
            custom_recurrent_output - custom_recurrent_ground_truth_1_output
        ).item()
        forward_error_2 = torch.norm(
            custom_recurrent_output - custom_recurrent_ground_truth_2_output
        ).item()
        forward_error_1_2 = torch.norm(
            custom_recurrent_output - custom_recurrent_ground_truth_2_output
        ).item()
        if verbose is True:
            print(f"Forward Error 1: {forward_error_1}")
            print(f"Forward Error 2: {forward_error_2}")
            print(f"Forward Error 1-2: {forward_error_1_2}")

        # Use the same random gradient at output and perform backprop from there
        grad_output = torch.rand_like(custom_recurrent_output)
        torch.autograd.backward(custom_recurrent_output, grad_output)
        torch.autograd.backward(
            custom_recurrent_ground_truth_1_output, grad_output
        )
        torch.autograd.backward(
            custom_recurrent_ground_truth_2_output, grad_output
        )

        recurrent_weight_error_1 = torch.norm(
            recurrent_synapse_list[0].weight.grad
            - recurrent_synapse_list[1].weight.grad
        ).item()
        recurrent_weight_error_2 = torch.norm(
            recurrent_synapse_list[0].weight.grad
            - recurrent_synapse_list[2].weight.grad
        ).item()
        recurrent_weight_error_1_2 = torch.norm(
            recurrent_synapse_list[1].weight.grad
            - recurrent_synapse_list[2].weight.grad
        ).item()

        if verbose is True:
            print(f"Recurrent Weight Grad Error 1: {recurrent_weight_error_1}")
            print(f"Recurrent Weight Grad Error 2: {recurrent_weight_error_2}")
            print(
                f"Recurrent Weight Grad Error 1-2: {recurrent_weight_error_1_2}"
            )

        current_decay_error_1 = torch.norm(
            neuron_list[0].current_decay.grad
            - neuron_list[1].current_decay.grad
        ).item()
        current_decay_error_2 = torch.norm(
            neuron_list[0].current_decay.grad
            - neuron_list[2].current_decay.grad
        ).item()
        if verbose is True:
            print(f"Current Decay Grad Error 1: {current_decay_error_1}")
            print(f"Current Decay Grad Error 2: {current_decay_error_2}")

        voltage_decay_error_1 = torch.norm(
            neuron_list[0].voltage_decay.grad
            - neuron_list[1].voltage_decay.grad
        ).item()
        voltage_decay_error_2 = torch.norm(
            neuron_list[0].voltage_decay.grad
            - neuron_list[2].voltage_decay.grad
        ).item()
        if verbose is True:
            print(f"Voltage Decay Grad Error 1: {voltage_decay_error_1}")
            print(f"Voltage Decay Grad Error 2: {voltage_decay_error_2}")

        epsilon = 1e-3
        self.assertTrue(
            forward_error_1 < epsilon,
            "Error in recurrent. Expected "
            + f"forward_error_1<{epsilon}."
            + f" Found {forward_error_1=}.",
        )
        self.assertTrue(
            forward_error_2 < epsilon,
            "Error in recurrent. Expected "
            + f"forward_error_2<{epsilon}."
            + f" Found {forward_error_2=}.",
        )

        self.assertTrue(
            recurrent_weight_error_1 < epsilon,
            "Error in recurrent. Expected "
            + f"recurrent_weight_error_1<{epsilon}."
            + f" Found {recurrent_weight_error_1=}.",
        )
        self.assertTrue(
            recurrent_weight_error_2 < epsilon,
            "Error in recurrent. Expected "
            + f"recurrent_weight_error_2<{epsilon}."
            + f" Found {recurrent_weight_error_2=}.",
        )

        self.assertTrue(
            current_decay_error_1 < epsilon,
            "Error in recurrent. Expected "
            + f"recurrent_weight_error_1<{epsilon}."
            + f" Found {current_decay_error_1=}.",
        )
        self.assertTrue(
            current_decay_error_2 < epsilon,
            "Error in recurrent. Expected "
            + f"recurrent_weight_error_1<{epsilon}."
            + f" Found {current_decay_error_2=}.",
        )

        self.assertTrue(
            voltage_decay_error_1 < epsilon,
            "Error in recurrent. Expected "
            + f"recurrent_weight_error_1<{epsilon}."
            + f" Found {voltage_decay_error_1=}.",
        )
        self.assertTrue(
            voltage_decay_error_2 < epsilon,
            "Error in recurrent. Expected "
            + f"recurrent_weight_error_2<{epsilon}."
            + f" Found {voltage_decay_error_2=}.",
        )
