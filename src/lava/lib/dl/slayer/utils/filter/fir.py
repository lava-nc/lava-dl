# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Finite Impulse Response filtering methods.
"""

import torch
import torch.nn.functional as F
import numpy as np
from .conv import conv


class FIR(torch.nn.Module):
    """Finite impulse response filter. The filters are not learnable. For
    learnable filter, use `FIRBank` with one filter.

    Parameters
    ----------
    fir_response : array
        Desired FIR response. If it is None, an exponentially decaying filter
        is initialized. Defaults to None.
    time_constant : float
        time constant of exponentially decaying filter. Defaults to 1.
    length : int
        length of the FIR filter to initialize. Defaults to 20.
    sampling_time : float
        sampling time of FIR filter. Defaults to 1.

    Attributes
    ----------
    filter : torch tensor
        impulse response of FIR filter.
    sampling_time : float
        sampling time of FIR filter.

    """
    def __init__(
        self,
        fir_response=None, time_constant=1, length=20, sampling_time=1
    ):
        super(FIR, self).__init__()
        if fir_response is not None:
            self.register_buffer('filter', torch.FloatTensor(fir_response))
        else:
            self.register_buffer(
                'filter',
                torch.FloatTensor(np.exp(-np.arange(length) / time_constant)),
            )
        self.sampling_time = sampling_time

    def forward(self, input):
        """
        """
        return conv(input, self.filter, self.sampling_time)


class FIRBank(torch.nn.Conv3d):
    """Finite impulse response filter bank. The filters are learnable.

    Parameters
    ----------
    num_filter : int
        number of FIR filters in the bank.
    filter_length : float
        time length of the filter.
    sampling_time : float
        sampling time of the filter. Defaults to 1.
    scale : float
        initialization scaling factor for filter. Defaults to 1.

    Attributes
    ----------
    sampling_time : float
        sampling time of the filter. Defaults to 1.

    """
    def __init__(self, num_filter, filter_length, sampling_time=1, scale=1):
        in_channels = 1
        out_channels = num_filter
        kernel = (1, 1, int(filter_length / sampling_time))

        super(FIRBank, self).__init__(
            in_channels, out_channels, kernel, bias=False
        )

        self.sampling_time = sampling_time
        self.pad = torch.nn.ConstantPad3d(
            padding=(filter_length - 1, 0, 0, 0, 0, 0),
            value=0,
        )

        if scale != 1:
            self.weight.data *= scale

    def forward(self, input):
        """
        """
        output_shape = [s for s in input.shape]
        output_shape[1] = -1
        return F.conv3d(
            self.pad(
                input.reshape((input.shape[0], 1, 1, -1, input.shape[-1]))
            ),
            self.weight
        ).reshape(output_shape) * self.sampling_time

    @property
    def num_filter(self):
        """Number of filters in the bank."""
        return self.out_channels

    @property
    def filter_length(self):
        """Time length of the filter."""
        return self.kernel_size[2]

    @property
    def impulse_response(self):
        """Impulse response of filter bank"""
        return self.weight.reshape(
            self.out_channels, -1
        ).cpu().data.numpy()[:, ::-1]
