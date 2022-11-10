# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Complex synapse"""

import torch
import numpy as np
from ..synapse import layer


class ComplexLayer(torch.nn.Module):
    """Abstract complex layer class.
    """
    def __init__(self):
        super(ComplexLayer, self).__init__()
        self.real = lambda x: x
        self.imag = lambda x: x
        self.complex = True

    def forward(self, input):
        """Forward complex synaptic operation.
        """
        return self.real(input), self.imag(input)

    def enable_weight_norm(self):
        """Enables weight normalization on synapse."""
        self.real.enable_weight_norm()
        self.imag.enable_weight_norm()

    def disable_weight_norm(self):
        """Disables weight normalization on synapse."""
        self.real.disable_weight_norm()
        self.imag.disable_weight_norm()

    @property
    def grad_norm(self):
        """Norm of weight gradients. Useful for monitoring gradient flow."""
        return np.sqrt(self.real.grad_norm**2 + self.imag.grad_norm**2)

    @property
    def pre_hook_fx(self):
        """Returns the pre-hook function for synapse operation. Typically
        intended to define the quantization method."""
        if id(self.real.preHookFx) != id(self.imag.preHookFx):
            raise Exception(
                'Quantization method of real and imaginary weights '
                'must be same.'
            )
        return self.real.preHookFx

    @pre_hook_fx.setter
    def pre_hook_fx(self, fx):
        """Sets the pre-hook function for synapse operation. Typically intended
        to define the quantization method."""
        self.real._pre_hook_fx = fx
        self.imag._pre_hook_fx = fx

    @property
    def shape(self):
        """Shape of the synapse"""
        return self.real.shape


class Dense(ComplexLayer):
    """Dense compelx-synapse layer.

    Parameters
    ----------
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    weight_scale : int
        weight initialization scaling factor. Defaults to 1.
    weight_norm : bool
        flag to enable/disable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function reference or a lambda function. If the function is provided,
        it will be applied to it's weight before the forward operation of the
        synapse. Typically the function is a quantization mechanism of the
        synapse. Defaults to None.

    Attributes
    ----------
    real : slayer.synapse.Dense
        real synapse.
    imag : slayer.synapse.Dense
        imaginary synapse.
    complex : bool
        True. Indicates synapse is complex.
    """
    def __init__(
        self,
        in_neurons, out_neurons,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        """ """
        super(Dense, self).__init__()
        self.real = layer.Dense(
            in_neurons, out_neurons,
            weight_scale, weight_norm, pre_hook_fx
        )
        self.imag = layer.Dense(
            in_neurons, out_neurons,
            weight_scale, weight_norm, pre_hook_fx
        )

        self.in_channels = self.real.in_channels
        self.out_channels = self.real.out_channels
        self.weight_norm_enabled = self.real.weight_norm_enabled


class Conv(ComplexLayer):
    """Convolution complex-synapse layer.

    Parameters
    ----------
    in_features : int
        number of input features.
    out_features : int
        number of output features.
    kernel_size : int or tuple of two ints
        size of the convolution kernel.
    stride : int or tuple of two ints
        stride of the convolution. Defaults to 1.
    padding : int or tuple of two ints
        padding of the convolution. Defaults to 0.
    dilation : int or tuple of two ints
        dilation of the convolution. Defaults to 1.
    groups : int
        number of blocked connections from input channel to output channel.
        Defaults to 1.
    weight_scale : int
        weight initialization scale factor. Defaults to 1.
    weight_norm : bool
        flag to enable/disable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function reference or a lambda function. If the function is provided,
        it will be applied to it's weight before the forward operation of the
        synapse. Typically the function is a quantization mechanism of the
        synapse. Defaults to None.

    Note
    ----
    For kernel_size, stride, padding and dilation, the tuple of two ints are
    represented in (height, width) order. The integer value is broadcast to
    height and width.

    Attributes
    ----------
    real : slayer.synapse.Conv
        real synapse.
    imag : slayer.synapse.Conv
        imaginary synapse.
    complex : bool
        True. Indicates synapse is complex.
    """
    def __init__(
        self, in_features, out_features, kernel_size,
        stride=1, padding=0, dilation=1, groups=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        """ """
        super(Conv, self).__init__()
        self.real = layer.Conv(
            in_features, out_features, kernel_size,
            stride, padding, dilation, groups,
            weight_scale, weight_norm, pre_hook_fx
        )
        self.imag = layer.Conv(
            in_features, out_features, kernel_size,
            stride, padding, dilation, groups,
            weight_scale, weight_norm, pre_hook_fx
        )


class Pool(ComplexLayer):
    """Pooling complex-synape layer.

    Parameters
    ----------
    kernel_size : int
        [description]
    stride : int or tuple of two ints
        stride of pooling. Defaults to `kernel_size`.
    padding : int or tuple of two ints
        padding of the pooling. Defaults to 0.
    dilation : int or tuple of two ints
        dilation of the pooling. Defaults to 1.
    weight_scale : int
        weight initialization scale factor. Defaults to 1.
    weight_norm : bool
        flag to enable/disable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function reference or a lambda function. If the function is provided,
        it will be applied to it's weight before the forward operation of the
        synapse. Typically the function is a quantization mechanism of the
        synapse. Defaults to None.

    Note
    ----
    For kernel_size, stride, padding and dilation, the tuple of two ints are
    represented in (height, width) order. The integer value is broadcast to
    height and width.

    Attributes
    ----------
    real : slayer.synapse.Pool
        real synapse.
    imag : slayer.synapse.Pool
        imaginary synapse.
    complex : bool
        True. Indicates synapse is complex.

    """
    def __init__(
        self, kernel_size,
        stride=None, padding=0, dilation=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        """ """
        super(Pool, self).__init__()
        self.real = layer.Pool(
            kernel_size, stride, padding, dilation,
            weight_scale, weight_norm, pre_hook_fx
        )
        self.imag = layer.Pool(
            kernel_size, stride, padding, dilation,
            weight_scale, weight_norm, pre_hook_fx
        )


class ConvTranspose(ComplexLayer):
    """Transposed convolution synapse layer.

    Parameters
    ----------
    in_features : int
        number of input features.
    out_features : int
        number of output features.
    kernel_size : int or tuple of two ints
        size of the transposed convolution kernel.
    stride : int or tuple of two ints
        stride of the transposed convolution. Defaults to 1.
    padding : int or tuple of two ints
        padding of the transposed convolution. Defaults to 0.
    dilation : int or tuple of two ints
        dilation of the transposed convolution. Defaults to 1.
    groups : int
        number of blocked connections from input channel to output channel.
        Defaults to 1.
    weight_scale : int
        weight initialization scale factor. Defaults to 1.
    weight_norm : bool
        flag to enable/disable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function reference or a lambda function. If the function is provided,
        it will be applied to it's weight before the forward operation of the
        synapse. Typically the function is a quantization mechanism of the
        synapse. Defaults to None.

    Note
    ----
    For kernel_size, stride, padding and dilation, the tuple of two ints are
    represented in (height, width) order. The integer value is broadcast to
    height and width.

    Attributes
    ----------
    real : slayer.synapse.ConvTranspose
        real synapse.
    imag : slayer.synapse.ConvTranspose
        imaginary synapse.
    complex : bool
        True. Indicates synapse is complex.

    """
    def __init__(
        self,
        in_features, out_features, kernel_size,
        stride=1, padding=0, dilation=1, groups=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        """ """
        super(ConvTranspose, self).__init__()
        self.real = layer.ConvTranspose(
            in_features, out_features, kernel_size,
            stride, padding, dilation, groups,
            weight_scale, weight_norm, pre_hook_fx
        )
        self.imag = layer.ConvTranspose(
            in_features, out_features, kernel_size,
            stride, padding, dilation, groups,
            weight_scale, weight_norm, pre_hook_fx
        )


class Unpool(ComplexLayer):
    """Unpooling complex-synape layer.

    Parameters
    ----------
    kernel_size : int
        [description]
    stride : int or tuple of two ints
        stride of unpooling. Defaults to `kernel_size`.
    padding : int or tuple of two ints
        padding of the unpooling. Defaults to 0.
    dilation : int or tuple of two ints
        dilation of the unpooling. Defaults to 1.
    weight_scale : int
        weight initialization scale factor. Defaults to 1.
    weight_norm : bool
        flag to enable/disable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function reference or a lambda function. If the function is provided,
        it will be applied to it's weight before the forward operation of the
        synapse. Typically the function is a quantization mechanism of the
        synapse. Defaults to None.

    Note
    ----
    For kernel_size, stride, padding and dilation, the tuple of two ints are
    represented in (height, width) order. The integer value is broadcast to
    height and width.

    Attributes
    ----------
    real : slayer.synapse.Unpool
        real synapse.
    imag : slayer.synapse.Unpool
        imaginary synapse.
    complex : bool
        True. Indicates synapse is complex.

    """
    def __init__(
        self,
        kernel_size,
        stride=None, padding=0, dilation=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        """ """
        super(Unpool, self).__init__()
        self.real = layer.Unpool(
            kernel_size, stride, padding, dilation,
            weight_scale, weight_norm, pre_hook_fx
        )
        self.imag = layer.Unpool(
            kernel_size, stride, padding, dilation,
            weight_scale, weight_norm, pre_hook_fx
        )
