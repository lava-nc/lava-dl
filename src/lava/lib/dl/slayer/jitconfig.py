# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Interface to change the behavior of jit compilation output.
"""

import torch

TORCH_CUDA_ARCH_LIST = None
VERBOSE = False

# following is temporary fix for new 3080 gpus until they get full support in
# torch
# TODO: remove when there is support
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        TORCH_CUDA_ARCH_LIST = '8.0'
