# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Integer bit shift utilities."""

import torch


def right_shift_to_zero(x, bits):
    """Right shift with quantization towards zero implementation.

    Parameters
    ----------
    x : torch.int32 or torch.int64
        input tensor.
    bits : int
        number of bits to shift.

    Returns
    -------
    torch.int32 or torch.int64
        right shift to zero result.

    """
    if not(x.dtype == torch.int32 or x.dtype == torch.int64):
        raise Exception(
            f'Expected torch.int32 or torch.int64 data, found {x.dtype}.'
        )
    x_sign = 2 * (x > 0) - 1
    # return x_sign * (x_sign * x >> bits) # This seems to return torch.int64!
    return (x_sign * ((x_sign * x) >> bits)).to(x.dtype)
    # return (x_sign * ((x_sign * x) / (1<<bits))).to(torch.int32)


class Q2Zero(torch.autograd.Function):
    """Autograd compliant version of quantization towards zero."""
    # fully autograd compliant version of right_shift_to_zero.
    # It is going to be slow. Just for verification purposes.
    @staticmethod
    def forward(ctx, x):
        """
        """
        x_sign = 2 * (x > 0) - 1
        return (x_sign * (x_sign * x).to(torch.int32)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        return grad_output
