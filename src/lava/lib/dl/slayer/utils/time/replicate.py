# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Tensor replication in time.
"""

import torch.nn.functional as F


def replicate(input, num_steps):
    """Replicates input in time dimension. Additional dimension of time is
    added at the end.

    Parameters
    ----------
    input : torch tensor
        torch input tensor.
    num_steps : int
        number of steps to replicate.

    Returns
    -------
    torch tensor
        input replicated num_steps times in time

    Examples
    --------

    >>> input = torch.rand(2, 3, 4)
    >>> out = replicate(input, 10)
    """
    ext_shape = [s for s in input.shape] + [1]
    out_shape = [s for s in input.shape] + [num_steps]
    return F.interpolate(
        input.reshape(ext_shape), size=out_shape[2:],
        mode='nearest',
    )
