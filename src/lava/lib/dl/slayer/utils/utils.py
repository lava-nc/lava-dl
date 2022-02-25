# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Collection of utilities."""

import torch


class staticproperty(property):
    """wraps static member function of a class as a static property of that
    class.
    """
    def __get__(self, cls, owner):
        return staticmethod(self.fget).__get__(None, owner)()


def diagonal_mask(dim, num_diagonals):
    """Creates a binary mask with ones around the major diagonal defined by
    `num_diagonals`.

    Parameters
    ----------
    dim : int
        dimension of the mask matrix
    num_diagonals : int
        number of diagonals. The number gets rounded up to the nearest odd
        number. 1 means an identity matrix.

    Returns
    -------
    torch tensor
        mask tensor

    """
    if num_diagonals == 0:
        raise Exception(
            f'Expected positive number of diagonals. Found {num_diagonals=}.'
        )
    mask = torch.eye(dim)
    for i in range(num_diagonals // 2):
        d = i + 1
        mask += torch.diag(torch.ones(dim - d), diagonal=d) \
            + torch.diag(torch.ones(dim - d), diagonal=-d)

    return mask
