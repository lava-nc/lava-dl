# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Dequantization utility."""

import torch
import numpy as np

def _dequantize(input):
    """Implements dequantization of parameters.

    Parameters
    ----------
    input : torch tensor
        input tensor

    Returns
    -------
    torch tensor
        dequantized tensor
    """    
    return torch.dequantize(input)

def dequantize(input):
    """Implements dequantization of parameters.

    Parameters
    ----------
    input : torch tensor
        input tensor

    Returns
    -------
    torch tensor
        dequantized tensor

    Examples
    --------
    >>> tensor = dequantize(x)
    """
    if type(input) is np.ndarray:
        tensor = torch.from_numpy(input)
        return _dequantize(tensor)
    else:
        return _dequantize(input)
