# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""HDF5 net description manipulation utilities."""

from typing import Tuple, Union
import h5py
import numpy as np
from enum import IntEnum, unique
import torch


@unique
class SYNAPSE_SIGN_MODE(IntEnum):
    """Enum for synapse sign mode. Options are {``MIXED : 1``,
    ``EXCITATORY : 2`` and ``INHIBITORY : 2``}.
    """
    MIXED = 1
    EXCITATORY = 2
    INHIBITORY = 3


class NetDict:
    """Provides dictionary like access to h5py object without the h5py quirks

    Parameters
    ----------
    filename : str or None, optional
        filename of h5py file to be loaded. It is only invoked if hdf5 file
        handle ``f`` is ``None``. Default is None.
    mode : str, optional
        file open mode, by default 'r'.
    f : h5py.File or h5py.Group, optional
        hdf5 file object handle. Overwrites the function of filename if it is
        not ``None``. Default is None.
    """

    def __init__(
        self,
        filename: Union[str, None] = None,
        mode: str = 'r',
        f: Union[h5py.File, h5py.Group] = None,
    ) -> None:
        self.f = h5py.File(filename, mode) if f is None else f
        self.str_keys = ['type']
        self.array_keys = [
            'shape', 'stride', 'padding', 'dilation', 'groups', 'delay',
            'iDecay', 'refDelay', 'scaleRho', 'tauRho', 'theta', 'vDecay',
            'vThMant', 'wgtExp', 'sinDecay', 'cosDecay'
        ]
        self.copy_keys = ['weight', 'bias', 'weight/real', 'weight/imag']

    def keys(self) -> h5py._hl.base.KeysViewHDF5:
        return self.f.keys()

    def __len__(self) -> int:
        return len(self.f)

    def __getitem__(self, key: str) -> h5py.Dataset:
        if key in self.str_keys:
            value = self.f[key]
            if len(value.shape) > 0:
                value = value[0]
            else:
                value = value[()]
            return value.decode('ascii')
        elif key in self.copy_keys:
            return self.f[key][()].astype(int).copy()
        elif key in self.array_keys:
            return self.f[key][()]
        elif isinstance(key, int) and f'{key}' in self.f.keys():
            return NetDict(f=self.f[f'{key}'])
        else:
            return NetDict(f=self.f[key])

    def __setitem__(self, key: str) -> None:
        raise NotImplementedError('Set feature is not implemented.')


def optimize_weight_bits(weight: np.ndarray) -> Tuple[
    np.ndarray, int, int, SYNAPSE_SIGN_MODE
]:
    """Optimizes the weight matrix to best fit in Loihi's synapse.

    Parameters
    ----------
    weight : np.ndarray
        standard 8 bit signed weight matrix.

    Returns
    -------
    np.ndarray
        optimized weight matrix
    int
        weight bits
    int
        weight_exponent
    SYNAPSE_SIGN_MODE
        synapse sign mode
    """
    max_weight = np.max(weight)
    min_weight = np.min(weight)

    if max_weight < 0:
        sign_mode = SYNAPSE_SIGN_MODE.INHIBITORY
        is_signed = 0
    elif min_weight >= 0:
        sign_mode = SYNAPSE_SIGN_MODE.EXCITATORY
        is_signed = 0
    else:
        sign_mode = SYNAPSE_SIGN_MODE.MIXED
        is_signed = 1

    if sign_mode == SYNAPSE_SIGN_MODE.MIXED:
        pos_scale = 127 / max_weight
        neg_scale = -128 / min_weight
        scale = np.min([pos_scale, neg_scale])
    elif sign_mode == SYNAPSE_SIGN_MODE.INHIBITORY:
        scale = -256 / min_weight
    elif sign_mode == SYNAPSE_SIGN_MODE.EXCITATORY:
        scale = 255 / max_weight

    scale_bits = np.floor(np.log2(scale)) + is_signed

    precision_found = False
    n = 8
    while (precision_found is False) and (n > 0):
        roundingError = np.sum(
            np.abs(weight / (2**n) - np.round(weight / (2**n)))
        )
        if roundingError == 0:
            precision_found = True
        else:
            n -= 1

    n -= is_signed

    num_weight_bits = 8 - scale_bits - n
    weight_exponent = -scale_bits

    weight = np.left_shift(weight.astype(np.int32), int(scale_bits))

    return (
        weight.astype(int),
        int(num_weight_bits),
        int(weight_exponent),
        sign_mode
    )


def num_delay_bits(delays: np.ndarray) -> int:
    """Calculates the number of delay bits required.

    Parameters
    ----------
    delays : np.ndarray
        delay vector

    Returns
    -------
    int
        number of delay bits.
    """
    if delays.min() < 0:
        raise ValueError(
            f'Negative delay encountered. '
            f'Found {delays.min()=}.'
        )
    if delays.max() >= 63:
        raise ValueError(
            f'Max delay exceeded limit of 62. '
            f'Found {delays.max()=}.'
        )
    return np.ceil(np.log2(delays.max())).astype(int)
