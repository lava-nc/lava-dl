# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Time shifts of tensor."""

import os
import numpy as np
import torch
from torch.utils.cpp_extension import load

from ..utils import staticproperty
from ... import jitconfig


class Accelerated:
    """ """
    module = None

    @staticproperty
    def shift():
        """ """
        if Accelerated.module is None:
            if jitconfig.VERBOSE is True:
                print(
                    'Shift accelerated module does not exist. '
                    'Initializing with JIT compilation'
                )
            if not torch.cuda.is_available():
                raise Exception(
                    'CUDA acceleration of time-shift failed. '
                    'CUDA is not available in the system.'
                )
            if jitconfig.TORCH_CUDA_ARCH_LIST is not None:
                os.environ['TORCH_CUDA_ARCH_LIST'] = \
                    jitconfig.TORCH_CUDA_ARCH_LIST
            Accelerated.module = load(
                name='shift',
                sources=[
                    os.path.dirname(os.path.abspath(__file__))
                    + '/shift.cu'
                ],
            )
        return Accelerated.module


def _shift_group(input, shift_val, sampling_time=1):
    """
    """
    if len(input.shape) <= 1:
        raise AssertionError(
            f'Expected input to have at least two dimension. '
            f'Found {input.shape=}.'
        )
    if hasattr(shift_val, 'grad'):
        shift_val = shift_val.item()
    shift_blocks = int(shift_val / sampling_time)
    out_shape = input.shape
    input = input.reshape(-1, input.shape[-1])
    output = torch.zeros_like(input)
    if shift_blocks == 0:
        output = input
    elif shift_blocks > 0:
        output = torch.zeros_like(input)
        output[..., shift_blocks:] = input[..., :-shift_blocks]
    else:
        output = torch.zeros_like(input)
        output[..., :shift_blocks] = input[..., -shift_blocks:]
    return output.reshape(out_shape)


def _shift_individual(input, shift_val, sampling_time=1):
    """
    """
    if np.prod(input.shape[1:-1]) != shift_val.numel():
        raise AssertionError(
            f'Expected spatial dimension of input and shift_val to be same. '
            f'Found {input.shape=}, {shift_val.shape=}.'
        )
    out_shape = input.shape
    input = input.reshape(input.shape[0], -1, input.shape[-1])
    output = torch.zeros_like(input)
    shift_val = shift_val.flatten() / sampling_time

    for i in range(output.shape[1]):
        output[:, i:i + 1] = _shift_group(
            input[:, i:i + 1],
            int(shift_val[i].item()),
            sampling_time
        )

    return output.reshape(out_shape)


def shift(input, shift_val, sampling_time=1):
    """Implements shift in time axis.

    Parameters
    ----------
    input : torch tensor
        input tensor.
    shift_val : torch tensor or float or int
        shift tensor. If it is scalar, same shift is
        applied to all the spatial dimension. Otherwise, the input's spatial
        dimension  must match shift's dimension.
    sampling_time : float
        sampling time. Defaults to 1.

    Returns
    -------
    torch tensor
        shifted output

    Examples
    --------

    >>> output = shift(input, 7)
    >>> output = shift(torch.rand(1, 10, 100), torch.arange(10))
    """
    if hasattr(shift_val, 'grad') and shift_val.numel() > 1:
        if np.prod(input.shape[1:-1]) != shift_val.numel():
            raise Exception(
                f'Expected spatial dimension of input and shift_val to be '
                f'same. Found {input.shape=}, {shift_val.shape=}.'
            )
        if input.is_cuda is False:
            return _shift_individual(input, shift_val, sampling_time)
        else:
            return Accelerated.shift.shift(
                input.contiguous(),
                shift_val.contiguous(),
                sampling_time
            )
    else:
        if hasattr(shift_val, 'grad'):
            shift_val = shift_val.item()
        if input.is_cuda is False:
            return _shift_group(input, shift_val, sampling_time)
        else:
            return Accelerated.shift.shift(
                input.contiguous(),
                shift_val,
                sampling_time
            )
