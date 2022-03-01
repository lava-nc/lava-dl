# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Methods for Convolution and Correlation in time.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from ..utils import staticproperty
from ... import jitconfig


class Accelerated:
    """
    """
    module = None

    @staticproperty
    def conv():
        """
        """
        if Accelerated.module is None:
            if jitconfig.VERBOSE is True:
                print(
                    'Conv accelerated module does not exist. '
                    'Initializing with JIT compilation'
                )
            if not torch.cuda.is_available():
                raise Exception(
                    'CUDA acceleration of time-conv failed. '
                    'CUDA is not available in the system.'
                )
            if jitconfig.TORCH_CUDA_ARCH_LIST is not None:
                os.environ['TORCH_CUDA_ARCH_LIST'] = \
                    jitconfig.TORCH_CUDA_ARCH_LIST
            Accelerated.module = load(
                name='conv',
                sources=[
                    os.path.dirname(os.path.abspath(__file__)) + '/conv.cu'
                ],
            )
        return Accelerated.module


def _fwd(input, filter, sampling_time=1):
    """
    """
    return F.conv3d(
        F.pad(
            input.reshape((input.shape[0], 1, 1, -1, input.shape[-1])),
            pad=(torch.numel(filter) - 1, 0, 0, 0, 0, 0),
            value=0,
        ),
        torch.flip(filter.reshape(1, 1, 1, 1, -1), dims=[-1])
    ).reshape(input.shape) * sampling_time


def _bwd(input, filter, sampling_time=1):
    """
    """
    return F.conv3d(
        F.pad(
            input.reshape((input.shape[0], 1, 1, -1, input.shape[-1])),
            pad=(0, torch.numel(filter) - 1, 0, 0, 0, 0),
            value=0,
        ),
        filter.reshape(1, 1, 1, 1, -1)
    ).reshape(input.shape) * sampling_time


def fwd(input, filter, sampling_time=1):
    """
    """
    if input.is_cuda is False:
        return _fwd(input, filter, sampling_time)
    else:
        return Accelerated.conv.fwd(
            input.contiguous(),
            filter.contiguous(),
            sampling_time
        )


def bwd(input, filter, sampling_time=1):
    """
    """
    if input.is_cuda is False:
        return _bwd(input, filter, sampling_time)
    else:
        return Accelerated.conv.bwd(
            input.contiguous(),
            filter.contiguous(),
            sampling_time
        )


class _conv(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx, input, filter, sampling_time):
        """
        """
        ctx.save_for_backward(
            filter,
            torch.autograd.Variable(
                torch.tensor(
                    sampling_time,
                    device=input.device,
                    dtype=input.dtype,
                ),
                requires_grad=False,
            ),
        )
        return fwd(input, filter, sampling_time)

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        filter, sampling_time = ctx.saved_tensors
        grad_input = bwd(grad_output, filter, sampling_time.item())
        if filter.requires_grad is False:
            grad_filter = None
        else:
            print('convolution filter gradient is not implemented.')
            grad_filter = None
        return grad_input, grad_filter, None


class _corr(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx, input, filter, sampling_time):
        """
        """
        ctx.save_for_backward(
            filter,
            torch.autograd.Variable(
                torch.tensor(
                    sampling_time,
                    device=input.device,
                    dtype=input.dtype
                ),
                requires_grad=False,
            ),
        )
        return bwd(input, filter, sampling_time)

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        filter, sampling_time = ctx.saved_tensors
        grad_input = fwd(grad_output, filter, sampling_time.item())
        if filter.requires_grad is False:
            grad_filter = None
        else:
            print('correlation filter gradient is not implemented.')
            grad_filter = None
        return grad_input, grad_filter, None


def conv(input, filter, sampling_time=1):
    """Convolution in time.

    Parameters
    ----------
    input : torch tensor
        input signal. Last dimension is assumed to be time.
    filter : torch tensor
        convolution filter. Assumed to be 1 dimensional. It will be flattened
        otherwise.
    sampling_time : float
        sampling time. Defaults to 1.

    Returns
    -------
    torch tensor
        convolution output. Output shape is same as input.

    Examples
    --------

    >>> output = conv(input, filter)
    """
    if len(input.shape) == 1:
        input = input.reshape(1, -1)

    return _conv.apply(input, filter.flatten(), sampling_time)


def corr(input, filter, sampling_time=1):
    """Correlation in time.

    Parameters
    ----------
    input : torch tensor
        input signal. Last dimension is assumed to be time.
    filter : torch tensor
        correlation filter. Assumed to be 1 dimensional. It will be flattened
        otherwise.
    sampling_time : float
        sampling time. Defaults to 1.

    Returns
    -------
    torch tensor
        correlation output. Output shape is same as input.

    Examples
    --------

    >>> output = corr(input, filter)
    """
    if len(input.shape) == 1:
        input = input.reshape(1, -1)

    return _corr.apply(input, filter.flatten(), sampling_time)
