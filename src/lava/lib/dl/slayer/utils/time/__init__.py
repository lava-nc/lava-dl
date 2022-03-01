# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Tensor time manipulation utilities."""

from .replicate import replicate
from .shift import shift

__all__ = ['replicate', 'shift']
