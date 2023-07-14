# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import unittest
import torch

import lava.lib.dl.slayer as slayer

verbose = True if (("-v" in sys.argv) or ("--verbose" in sys.argv)) else False


class TestTensorToEventWithTorchInput(unittest.TestCase):
    def test_torch_input(self):

        if verbose is True:
            print("Testing torch tensor input")

        tensor = torch.rand(2, 10, 10, 5)
        tensor[tensor > 0.5] = 0

        self.assertIsInstance(slayer.io.tensor_to_event(tensor),
                              slayer.io.Event, msg="Result is not an event")

    def test_identical_output(self):

        if verbose is True:
            print("Testing if numpy and torch inputs provide identical output")

        tensor = torch.rand(2, 10, 10, 5)
        tensor[tensor > 0.5] = 0

        event_tensor = slayer.io.tensor_to_event(tensor)
        event_numpy = slayer.io.tensor_to_event(tensor.numpy())

        epsilon = 1e-9
        self.assertTrue((event_tensor.t == event_numpy.t).all(),
                        msg="time data does not match")
        self.assertTrue((event_tensor.x == event_numpy.x).all(),
                        msg="x data does not match")
        self.assertTrue((event_tensor.y == event_numpy.y).all(),
                        msg="y data does not match")
        self.assertTrue((event_tensor.c == event_numpy.c).all(),
                        msg="c data does not match")
        self.assertTrue((event_tensor.t - event_numpy.t).sum() < epsilon,
                        msg="time data does not match")

    def test_to_tensor_functionality(self):
        if verbose is True:
            print("Testing to_tensor() functionality")

        tensor = torch.rand(2, 10, 10, 5)
        tensor[tensor > 0.5] = 0

        event_tensor = slayer.io.tensor_to_event(tensor)
        event_numpy = slayer.io.tensor_to_event(tensor.numpy())

        epsilon = 1e-6
        tensor_from_tensor = event_tensor.to_tensor()
        tensor_from_numpy = event_numpy.to_tensor()
        self.assertTrue((tensor_from_tensor - tensor_from_numpy).sum() < epsilon
                        , msg='to_tensor() does not return same data')

    def test_fill_tensor_functionality(self):
        if verbose is True:
            print("Testing fill_tensor() functionality")

        tensor = torch.rand(2, 10, 10, 5)
        tensor[tensor > 0.5] = 0

        event_tensor = slayer.io.tensor_to_event(tensor)
        event_numpy = slayer.io.tensor_to_event(tensor.numpy())

        epsilon = 1e-6
        fill_from_tensor = event_tensor.fill_tensor(torch.zeros(2, 10, 10, 5))
        fill_from_numpy = event_numpy.fill_tensor(torch.zeros(2, 10, 10, 5))
        self.assertTrue((fill_from_tensor - fill_from_numpy).sum().item()
                        < epsilon, msg='fill_tensor() does not return same data'
                        )


if __name__ == '__main__':
    unittest.main()
