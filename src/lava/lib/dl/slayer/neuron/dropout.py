# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Neuron Dropout."""

import torch
import torch.nn.functional as F


class Dropout(torch.nn.Dropout3d):
    """Neuron dropout method. It behaves similar to `torch.nn.Dropout`.
    However, dropout over time dimension is preserved, i.e. if a neuron is
    dropped, it remains dropped for the entire time duration.

    Parameters
    ----------
    p : float
        dropout probability.
    inplace : bool
        inplace operation flag. Default is False.

    Examples
    --------

    >>> drop = Dropout(0.2, inplace=True)
    >>> output = drop(input)
    """
    def forward(self, input):
        """
        """
        input_shape = input.shape
        return F.dropout3d(
            input.reshape((input_shape[0], -1, 1, 1, input_shape[-1])),
            self.p, self.training, self.inplace
        ).reshape(input_shape)
