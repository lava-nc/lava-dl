# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Axon delay implementation."""

import torch
import numpy as np
from ..utils.time import shift
from ..utils.filter import conv


def delay(input, delay_val=1, sampling_time=1):
    """Delay the signal in time.

    Parameters
    ----------
    input : torch.tensor
        Input signal. The last dimension is assumed to be time dimension.
    delay : int
        Amount of delay to apply. Defaults to 1.
    sampling_time : int
        Sampling time of delay operation. Defaults to 1.

    Returns
    -------
    torch.tensor
        delayed signal

    Examples
    --------

    >>> x_delayed = delay(x, 2) # delay x by 2 timesteps

    """
    return _delayFunctionNoGradient.apply(input, delay_val, sampling_time)


class Delay(torch.nn.Module):
    """Learnable axonal delay module. The delays operate on channel dimension.

    Parameters
    ----------
    sampling_time : int
        Sampling time of delay. Defaults to 1.
    max_delay : int
        Maximum allowable delay. Defaults to None.
    grad_scale : float
        gradient scale parameter. Defaults to 1.

    Attributes
    ----------
    sampling_time
    max_delay
    grad-scale
    delay : torch parameter
        the delay parameter.

    Examples
    --------
    >>> axon_delay = Delay()
    >>> x_delayed = axon_delay(x)
    """
    # It only operates on channel dimension, by design
    # def __init__(self, sampling_time=1, max_delay=None, grad_scale=(1<<12)):
    def __init__(self, sampling_time=1, max_delay=None, grad_scale=1):
        super(Delay, self).__init__()
        self.sampling_time = sampling_time
        self.max_delay = max_delay
        self.grad_scale = grad_scale
        # self.register_parameter('delay', None)
        self.register_parameter(
            'delay',
            torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        )
        self.init = False

        self.clamp()

    def clamp(self):
        """Clamps delay to allowable range. Typically it is not needed to be
        called explicitly."""
        if self.init is True:
            if self.max_delay is None:
                self.delay.data.clamp_(0)
            else:
                self.delay.data.clamp_(0, self.max_delay)

    @property
    def shape(self):
        """Shape of the delay."""
        if self.init is False:
            return None
        return self.delay.shape

    def forward(self, input):
        """Apply delay to input tensor.

        Parameters
        ----------
        input : torch.tensor
            input tensor.

        Returns
        -------
        torch.tensor
            delayed tensor.

        """
        if self.init is False:
            if len(input.shape) <= 2:
                raise AssertionError(
                    f"Expected input to have at least 3 dimensions: "
                    f"[Batch, Spatial dims ..., Time]. "
                    f"It's shape is {input.shape}."
                )

            self.delay.data = torch.rand(
                input.shape[1],
                dtype=torch.float,
                device=input.device
            )
            self.init = True

        self.clamp()

        if self.shape[0] != input.shape[1]:
            raise AssertionError(
                f"Expected input to have same number of channels as delays. "
                f"Expected {self.shape[0]}, found {input.shape[1]}."
            )

        if np.prod(input.shape[1:-1]) == self.shape[0]:
            return _delayFunction.apply(
                input,
                self.delay,
                self.grad_scale,
                self.sampling_time
            )
        else:
            broadcast_shape = list(input.shape[1:-1])
            broadcast_shape[0] = 1
            return _delayFunction.apply(
                input,
                self.delay.reshape(-1, 1, 1).repeat(broadcast_shape),
                self.grad_scale,
                self.sampling_time
            )


class _delayFunctionNoGradient(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(ctx, input, delay_val, sampling_time):
        """
        """
        device = input.device
        dtype = input.dtype
        output = shift(input, delay_val, sampling_time)
        sampling_time = torch.autograd.Variable(
            torch.tensor(sampling_time, device=device, dtype=dtype),
            requires_grad=False
        )
        delay_val = torch.autograd.Variable(
            torch.tensor(delay_val, device=device, dtype=dtype),
            requires_grad=False
        )
        ctx.save_for_backward(delay_val, sampling_time)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        (delay_val, sampling_time) = ctx.saved_tensors
        return shift(grad_output, -delay_val, sampling_time), None, None


class _delayFunction(torch.autograd.Function):
    """ """
    @staticmethod
    def forward(ctx, input, delay_val, grad_scale, sampling_time):
        """
        """
        device = input.device
        dtype = input.dtype
        output = shift(input, delay_val.data, sampling_time)
        sampling_time = torch.autograd.Variable(
            torch.tensor(sampling_time, device=device, dtype=dtype),
            requires_grad=False
        )
        grad_scale = torch.autograd.Variable(
            torch.tensor(grad_scale, device=device, dtype=dtype),
            requires_grad=False
        )
        ctx.save_for_backward(
            output, delay_val.data, grad_scale, sampling_time
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        (output, delay_val, grad_scale, sampling_time) = ctx.saved_tensors
        diff_filter = torch.tensor(
            [-1, 1], dtype=grad_output.dtype
        ).to(grad_output.device) / sampling_time
        output_diff = conv(output, diff_filter, 1)
        # the conv operation should not be scaled by sampling_time.
        # As such, the output is -( x[k+1]/sampling_time - x[k]/sampling_time )
        # which is what we want.
        grad_delay = torch.sum(
            grad_output * output_diff,
            [0, -1],
            keepdim=True
        ).reshape(grad_output.shape[1:-1]) * sampling_time
        # no minus needed here, as it is included in diff_filter
        # which is -1 * [1, -1]

        return shift(grad_output, -delay_val, sampling_time), \
            grad_delay * grad_scale, None, None
