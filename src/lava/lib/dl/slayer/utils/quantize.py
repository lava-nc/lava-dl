# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Quantization utility."""

import torch
from enum import IntEnum, unique


@unique
class MODE(IntEnum):
    """Quantization mode constants. Options are {``ROUND : 0``, ``FLOOR : 1``}.
    """
    ROUND = 0
    FLOOR = 1


class _quantize(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(ctx, input, step=1):
        """
        """
        # return input
        # print('input quantized with step', step)
        return torch.round(input / step) * step

    @staticmethod
    def backward(ctx, gradOutput):
        """
        """
        return gradOutput, None


class _floor(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(ctx, input, step=1):
        """
        """
        # return input
        # print('input quantized with step', step)
        return torch.floor(input / step) * step

    @staticmethod
    def backward(ctx, gradOutput):
        """
        """
        return gradOutput, None


def quantize(input, step=1, mode=MODE.ROUND):
    """Implements quantization of parameters. Round or floor behavior can be
    selected using mode argument.

    Parameters
    ----------
    input : torch tensor
        input tensor
    step : float
        quantization step. Default is 1.
    mode : MODE
        quantization mode. Default is MODE.ROUND.

    Returns
    -------
    torch tensor
        quantized tensor

    Examples
    --------

    >>> # Quantize in step of 0.5
    >>> x_quantized = quantize(x, step=0.5)
    """
    if mode == MODE.ROUND:
        return _quantize.apply(input, step)
    elif mode == MODE.FLOOR:
        return _floor.apply(input, step)
    else:
        raise ValueError(f'{mode=} is not recognized.')
