# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Synapse module"""

import numpy as np
import torch
import torch.nn.functional as F


class GenericLayer(torch.nn.Module):
    """Abstract synapse layer class.

    Attributes
    ----------
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(self):
        super(GenericLayer, self).__init__()
        self.weight_norm_enabled = False
        self.complex = False

    def enable_weight_norm(self):
        """Enables weight normalization on synapse."""
        self = torch.nn.utils.weight_norm(self, name='weight')
        self.weight_norm_enabled = True

    def disable_weight_norm(self):
        """Disables weight normalization on synapse."""
        torch.nn.utils.remove_weight_norm(self, name='weight')
        self.weight_norm_enabled = False

    @property
    def grad_norm(self):
        """Norm of weight gradients. Useful for monitoring gradient flow."""
        if self.weight_norm_enabled is False:
            if self.weight.grad is None:
                return 0
            else:
                return torch.norm(
                    self.weight.grad
                ).item() / torch.numel(self.weight.grad)
        else:
            if self.weight_g.grad is None:
                return 0
            else:
                return torch.norm(
                    self.weight_g.grad
                ).item() / torch.numel(self.weight_g.grad)

    @property
    def pre_hook_fx(self):
        """Returns the pre-hook function for synapse operation. Typically
        intended to define the quantization method."""
        return self._pre_hook_fx

    @pre_hook_fx.setter
    def pre_hook_fx(self, fx):
        """Sets the pre-hook function for synapse operation. Typically intended
        to define the quantization method.
        """
        self._pre_hook_fx = fx

    @property
    def shape(self):
        """Shape of the synapse"""
        return self.weight.shape


class Dense(torch.torch.nn.Conv3d, GenericLayer):
    """Dense synapse layer.

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
    in_channels
    out_channels
    weight
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(
        self,
        in_neurons, out_neurons,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        """ """
        # extract information for kernel and in_channels
        if type(in_neurons) == int:
            kernel = (1, 1, 1)
            in_channels = in_neurons
        elif len(in_neurons) == 2:
            kernel = (in_neurons[1], in_neurons[0], 1)
            in_channels = 1
        elif len(in_neurons) == 3:
            kernel = (in_neurons[1], in_neurons[0], 1)
            in_channels = in_neurons[2]
        else:
            raise Exception(
                f'in_neurons should not be more than 3 dimension. '
                f'Found {in_neurons.shape=}'
            )

        if type(out_neurons) == int:
            out_channels = out_neurons
        else:
            raise Exception(
                f'out_neurons should not be more than 1 dimension. '
                f'Found {out_neurons.shape}'
            )

        super(Dense, self).__init__(
            in_channels, out_channels, kernel, bias=False
        )

        if weight_scale != 1:
            self.weight = torch.nn.Parameter(weight_scale * self.weight)

        self._pre_hook_fx = pre_hook_fx

        if weight_norm is True:
            self.enable_weight_norm()

    def forward(self, input):
        """Applies the synapse to the input.

        Parameters
        ----------
        input : torch tensor
            Input tensor. Typically spikes. Input is expected to be of shape
            NCT or NCHWT.

        Returns
        -------
        torch tensor
            dendrite accumulation / weighted spikes.

        """
        if self._pre_hook_fx is None:
            weight = self.weight
        else:
            weight = self._pre_hook_fx(self.weight)

        if len(input.shape) == 3:
            old_shape = input.shape
            return F.conv3d(  # bias does not need pre_hook_fx. Its disabled
                input.reshape(old_shape[0], -1, 1, 1, old_shape[-1]),
                weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            ).reshape(old_shape[0], -1, old_shape[-1])
        else:
            return F.conv3d(
                input, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )


class Conv(torch.nn.Conv3d, GenericLayer):
    """Convolution synapse layer.

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
    in_channels
    out_channels
    kernel
    stride
    padding
    dilation
    groups
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(
        self, in_features, out_features, kernel_size,
        stride=1, padding=0, dilation=1, groups=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        """ """
        in_channels = in_features
        out_channels = out_features

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception(
                f'kernel_size can only be of 1 or 2 dimension. '
                f'Found {kernel_size.shape=}'
            )

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception(
                f'stride can be either int or tuple of size 2. '
                f'Found {stride.shape=}'
            )

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception(
                f'padding can be either int or tuple of size 2. '
                f'Found {padding.shape=}'
            )

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception(
                f'dilation can be either int or tuple of size 2. '
                f'Found {dilation.shape=}'
            )

        # groups
        # no need to check for groups. It can only be int

        super(Conv, self).__init__(
            in_channels, out_channels,
            kernel, stride, padding, dilation, groups, bias=False
        )

        if weight_scale != 1:
            self.weight = torch.nn.Parameter(weight_scale * self.weight)

        self._pre_hook_fx = pre_hook_fx

        if weight_norm is True:
            self.enable_weight_norm()

    def forward(self, input):
        """Applies the synapse to the input.

        Parameters
        ----------
        input : torch tensor
            Input tensor. Typically spikes. Input is expected to be of shape
            NCHWT.

        Returns
        -------
        torch tensor
            dendrite accumulation / weighted spikes.

        """
        if self._pre_hook_fx is None:
            return F.conv3d(
                input,
                self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )
        else:
            return F.conv3d(
                input,
                self._pre_hook_fx(self.weight), self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )


class Pool(torch.nn.Conv3d, GenericLayer):
    """Pooling synape layer.

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
    in_channels
    out_channels
    weight
    kernel_size
    stride
    padding
    dilation
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(
        self, kernel_size,
        stride=None, padding=0, dilation=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        """ """
        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception(
                f'kernel_size can only be of 1 or 2 dimension. '
                f'Found {kernel_size.shape=}'
            )

        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception(
                f'stride can be either int or tuple of size 2. '
                f'Found {stride.shape=}'
            )

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception(
                f'padding can be either int or tuple of size 2. '
                f'Found {padding.shape=}'
            )

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception(
                f'dilation can be either int or tuple of size 2. '
                f'Found {dilation.shape=}'
            )

        super(Pool, self).__init__(
            1, 1,
            kernel, stride, padding, dilation, bias=False
        )

        self.weight = torch.nn.Parameter(
            torch.FloatTensor(
                weight_scale * np.ones((self.weight.shape))
            ).to(self.weight.device),
            requires_grad=False
        )

        self._pre_hook_fx = pre_hook_fx

        if weight_norm is True:
            self.enable_weight_norm()

    def forward(self, input):
        """Applies the synapse to the input.

        Parameters
        ----------
        input : torch tensor
            Input tensor. Typically spikes. Input is expected to be of shape
            NCHWT.

        Returns
        -------
        torch tensor
            dendrite accumulation / weighted spikes.

        """
        device = input.device
        dtype = input.dtype

        # add necessary padding for odd spatial dimension
        if input.shape[2] % self.weight.shape[2] != 0:
            input = torch.cat((
                input,
                torch.zeros((
                    input.shape[0], input.shape[1],
                    input.shape[2] % self.weight.shape[2],
                    input.shape[3], input.shape[4]
                ), dtype=dtype).to(device)), 2
            )
        if input.shape[3] % self.weight.shape[3] != 0:
            input = torch.cat((
                input,
                torch.zeros((
                    input.shape[0], input.shape[1], input.shape[2],
                    input.shape[3] % self.weight.shape[3],
                    input.shape[4]
                ), dtype=dtype).to(device)), 3
            )

        in_shape = input.shape

        if self._pre_hook_fx is None:
            result = F.conv3d(
                input.reshape((
                    in_shape[0],
                    1,
                    in_shape[1] * in_shape[2],
                    in_shape[3],
                    in_shape[4]
                )),
                self.weight, self.bias,
                self.stride, self.padding, self.dilation,
            )
        else:
            result = F.conv3d(
                input.reshape((
                    in_shape[0],
                    1,
                    in_shape[1] * in_shape[2],
                    in_shape[3],
                    in_shape[4]
                )),
                self._pre_hook_fx(self.weight), self.bias,
                self.stride, self.padding, self.dilation,
            )
        return result.reshape((
            result.shape[0],
            in_shape[1],
            -1,
            result.shape[3],
            result.shape[4]
        ))


class ConvTranspose(torch.nn.ConvTranspose3d, GenericLayer):
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
    in_channels
    out_channels
    weight
    kernel_size
    stride
    padding
    dilation
    groups
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(
        self, in_features, out_features, kernel_size,
        stride=1, padding=0, dilation=1, groups=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        in_channels = in_features
        out_channels = out_features

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception(
                f'kernel_size can only be of 1 or 2 dimension. '
                f'Found {kernel_size.shape=}'
            )

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception(
                f'stride can be either int or tuple of size 2. '
                f'Found {stride.shape=}'
            )

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception(
                f'padding can be either int or tuple of size 2. '
                f'Found {padding.shape=}'
            )

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception(
                f'dilation can be either int or tuple of size 2. '
                f'Found {dilation.shape=}'
            )

        # groups
        # no need to check for groups. It can only be int

        super(ConvTranspose, self).__init__(
            in_channels, out_channels,
            kernel, stride, padding, 0, groups, False, dilation,
        )

        if weight_scale != 1:
            self.weight = torch.nn.Parameter(weight_scale * self.weight)

        self._pre_hook_fx = pre_hook_fx

        if weight_norm is True:
            self.enable_weight_norm()

    def forward(self, input):
        """Applies the synapse to the input.

        Parameters
        ----------
        input : torch tensor
            Input tensor. Typically spikes. Input is expected to be of shape
            NCHWT.

        Returns
        -------
        torch tensor
            dendrite accumulation / weighted spikes.

        """
        if self._pre_hook_fx is None:
            return F.conv_transpose3d(
                input,
                self.weight, self.bias,
                self.stride, self.padding, self.output_padding,
                self.groups, self.dilation,
            )
        else:
            return F.conv_transpose3d(
                input,
                self._pre_hook_fx(self.weight), self.bias,
                self.stride, self.padding, self.output_padding,
                self.groups, self.dilation,
            )


class Unpool(torch.nn.ConvTranspose3d, GenericLayer):
    """Unpooling synape layer.

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
    in_channels
    out_channels
    weight
    stride
    padding
    dilation
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(
        self, kernel_size,
        stride=None, padding=0, dilation=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception(
                f'kernel_size can only be of 1 or 2 dimension. '
                f'Found {kernel_size.shape=}'
            )

        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception(
                f'stride can be either int or tuple of size 2. '
                f'Found {stride.shape=}'
            )

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception(
                f'padding can be either int or tuple of size 2. '
                f'Found {padding.shape=}'
            )

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception(
                f'dilation can be either int or tuple of size 2. '
                f'Found {dilation.shape=}'
            )

        super(Unpool, self).__init__(
            1, 1, kernel, stride, padding, 0, 1, False, dilation
        )

        self.weight = torch.nn.Parameter(
            torch.FloatTensor(
                weight_scale * np.ones((self.weight.shape))
            ).to(self.weight.device),
            requires_grad=False,
        )

        self._pre_hook_fx = pre_hook_fx

        if weight_norm is True:
            self.enable_weight_norm()

    def forward(self, input):
        """Applies the synapse to the input.

        Parameters
        ----------
        input : torch tensor
            Input tensor. Typically spikes. Input is expected to be of shape
            NCHWT.

        Returns
        -------
        torch tensor
            dendrite accumulation / weighted spikes.

        """
        # device = input.device
        # dtype  = input.dtype
        # # add necessary padding for odd spatial dimension
        # This is not needed as unpool multiplies the spatial dimension,
        # hence it is always fine
        # if input.shape[2]%self.weight.shape[2] != 0:
        #     input = torch.cat(
        #         (
        #             input,
        #             torch.zeros(
        #                 (input.shape[0],
        #                 input.shape[1], input.shape[2]%self.weight.shape[2],
        #                 input.shape[3], input.shape[4]),
        #                 dtype=dtype
        #             ).to(device)
        #         ),
        #         dim=2,
        #     )
        # if input.shape[3]%self.weight.shape[3] != 0:
        #     input = torch.cat(
        #         (
        #             input,
        #             torch.zeros(
        #                 (input.shape[0],
        #                 input.shape[1], input.shape[2],
        #                 input.shape[3]%self.weight.shape[3], input.shape[4]),
        #                 dtype=dtype
        #             ),
        #             dim=3,
        #         )
        #     )

        in_shape = input.shape

        if self._pre_hook_fx is None:
            result = F.conv_transpose3d(
                input.reshape((in_shape[0], 1, -1, in_shape[3], in_shape[4])),
                self.weight, self.bias,
                self.stride, self.padding, self.output_padding,
                self.groups, self.dilation,
            )
        else:
            result = F.conv_transpose3d(
                input.reshape((in_shape[0], 1, -1, in_shape[3], in_shape[4])),
                self._pre_hook_fx(self.weight), self.bias,
                self.stride, self.padding, self.output_padding,
                self.groups, self.dilation,
            )

        return result.reshape((
            result.shape[0],
            in_shape[1],
            -1,
            result.shape[3],
            result.shape[4]
        ))
