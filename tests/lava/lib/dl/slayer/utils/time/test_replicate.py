# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import unittest
import torch

from lava.lib.dl.slayer.utils.time.replicate import replicate

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False


class TestReplicate(unittest.TestCase):
    def test_replicate(self):
        num_steps = 10
        input = torch.rand(2, 3, 4)
        output = replicate(input, num_steps)
        error = 0
        for i in range(num_steps):
            error += torch.abs(output[..., i] - input).sum().item()

        if verbose is True:
            print(f'{input=}')
            print(f'{output[..., 0]=}')
            print(f'{output[..., 1]=}')
            print(f'{output[..., 2]=}')
        if error > 1e-6:
            raise Exception(
                f'Error in replication. Expected error<1e-6. Found {error=}.'
            )
