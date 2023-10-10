# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import unittest
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax

import lava.lib.dl.slayer as slayer

torch.manual_seed(0)


def calculate_timestep_rates(spikes: torch.Tensor, window: int) -> torch.Tensor:
    """
    Calculate the rate at every timestep using convolution.
    """
    # Number of channels in the spikes tensor
    channels = spikes.size(1)

    # Define the convolution kernel
    kernel = torch.ones(channels, 1, window).to(spikes.device) / window

    # Convolve the spikes tensor with the kernel using depthwise convolution
    rates = F.conv1d(
        spikes, kernel, padding=window - 1, groups=channels
    )[:, :, :-(window - 1)]

    return rates


# Initialize a 5 class spike train (this could come from an SNN)
num_classes = 5
num_samples = 2
timesteps = 60
input_data = torch.randint(0, 2, (num_samples, num_classes, timesteps)).float()
alpha = .01
theta = .5  # detection threshold
moving_window = 10
rates = calculate_timestep_rates(input_data, moving_window)


# Labels for Testing
class TestSpikemoid(unittest.TestCase):
    def test_global_2d_target(self):
        labels_2d = torch.zeros((num_samples, num_classes))
        global_probs = torch.sigmoid((input_data.mean(-1) - theta) / alpha)
        spikemoid = slayer.loss.SpikeMoid(
            reduction="mean", alpha=alpha, theta=theta)
        spikemoid_loss = spikemoid(input_data, labels_2d).float()
        manual_loss = F.binary_cross_entropy(global_probs, labels_2d)
        self.assertAlmostEqual(spikemoid_loss.item(), manual_loss.item(), 4)

    def test_sliding_2d_target(self):
        probs = torch.sigmoid((rates - theta) / alpha)
        labels_2d = torch.zeros((num_samples, num_classes))
        labels_3d = torch.zeros(probs.shape)
        spikemoid = slayer.loss.SpikeMoid(
            reduction="mean", alpha=alpha, theta=theta,
            moving_window=moving_window)
        spikemoid_loss = spikemoid(input_data, labels_2d)
        manual_loss = F.binary_cross_entropy(
            probs.flatten().float(), labels_3d.flatten(), reduction="mean")
        self.assertAlmostEqual(spikemoid_loss.item(), manual_loss.item(), 4)

    def test_sliding_3d_target(self):
        probs = torch.sigmoid((rates - theta) / alpha)
        labels_3d = torch.ones(probs.shape)
        spikemoid = slayer.loss.SpikeMoid(
            reduction="mean", alpha=alpha, theta=theta,
            moving_window=moving_window)
        spikemoid_loss = spikemoid(input_data, labels_3d)
        manual_loss = F.binary_cross_entropy(
            probs.flatten().float(), labels_3d.flatten(), reduction="mean")
        self.assertAlmostEqual(spikemoid_loss.item(), manual_loss.item(), 4)
