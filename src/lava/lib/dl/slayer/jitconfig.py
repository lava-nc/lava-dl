# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Interface to change the behavior of jit compilation output.
"""

import torch

TORCH_CUDA_ARCH_LIST = None
VERBOSE = False

# This code checks the CUDA compute capability
# TODO: As new NVIDIA GPUs are release, the major version check should be
#       updated to reflect support.
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    TORCH_CUDA_ARCH_LIST = f'{major}.{minor}'
    if major > 12:
        print('CUDA Compute Capability > 12 may not work.')
