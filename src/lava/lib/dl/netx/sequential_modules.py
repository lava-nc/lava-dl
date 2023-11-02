# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Sequential modules:
They are intended to quickly prototype modules that do their forward computation
as fast as possible and any slow management routines in between calls in a
separate parallel thread.

The modules have:
- `time_step` counter which can be used to time periodic calls managements.
- `forward()` function: It is responsible for all the computation that
  is done when the object is called.
- Optionally a `post_forward()` function: If defined, it will be launched
  in a separate thread without holding the the subsequent computation after
  the forward call. Example use case: loading data from disk between calls.

An example module:
.. code-block:: python
    class DelayBuffer(AbstractSeqModule):
        def __init__(self, shape: Tuple[int, ...], delay: int) -> None:
            super().__init__()
            self.buffer = np.zeros((*shape, delay))

        def __call__(self, inp: np.ndarray) -> np.ndarray:
            # This is only needed for proper type hinting.
            # It is a good practice.
            return super().__call__(inp=inp)

        def forward(self, inp: np.ndarray) -> np.ndarray:
            out = self.buffer[..., 0].copy()
            self.buffer[..., :-1] = self.buffer[..., 1:]
            self.buffer[..., -1] = inp
            return out
"""

import atexit
from typing import Tuple, Union, Any
from threading import Thread
from collections import deque

import numpy as np
import torch


class AbstractSeqModule:
    """This is an abstract template class for sequential modules.

    Any inherited class needs to define it's own
    - `forward()` function: It is responsible for all the computation that
      is done when the object is called.
    - Optionally a `post_forward()` function: If defined, it will be launched
      in a separate thread without holding the the subsequent computation after
      the forward call. Example use case: loading data from disk between calls.
    """

    def __init__(self) -> None:
        self.time_step = 0
        self.has_post_fx = hasattr(self, 'post_forward')
        self.post_fx_thread = None
        # Make sure to join all hanging threads at cleanup
        atexit.register(self._cleanup)

    def _cleanup(self):
        self._join_post_fx_thread()

    def _join_post_fx_thread(self):
        if self.post_fx_thread is not None:
            self.post_fx_thread.join()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.time_step += 1
        self._join_post_fx_thread()

        fwd_fx = getattr(self, 'forward')
        result = fwd_fx(*args, **kwds)
        if self.has_post_fx:
            self.post_fx_thread = Thread(target=getattr(self, 'post_forward'))
            self.post_fx_thread.start()
        return result


class Quantize(AbstractSeqModule):
    """Data quantization module.

    Parameters
    ----------
    exp : int, optional
        Number of fractional bits in fixed precision, by default 0.
    """

    def __init__(self, exp=0) -> None:
        super().__init__()
        self.exp = exp

    def __call__(self, x: Union[np.ndarray, torch.tensor]) -> np.ndarray:
        return super().__call__(x)

    def forward(self, x: Union[np.ndarray, torch.tensor]) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.cpu().data.numpy()
        return np.round(x * (1 << self.exp)).astype(int)


class Dequantize(AbstractSeqModule):
    """Data dequantization module.

    Parameters
    ----------
    exp : int, optional
        Number of fractional bits in fixed precision, by default 0.
    num_raw_bits : int, optional
        Number of bits in raw representation of the data, by default 24.

    Raises
    ------
    RuntimeError
        When `num_raw_bits` exceeds 32.
    """

    def __init__(self, exp=0, num_raw_bits=24) -> None:
        super().__init__()
        self.exp = exp
        self.shift_32 = 32 - num_raw_bits
        if self.shift_32 < 0:
            raise RuntimeError("This module is only capable of reinterpreting "
                               "num_raw_bits <= 32")

    def __call__(self, x: Union[np.ndarray, torch.tensor]) -> np.ndarray:
        return super().__call__(x)

    def forward(self, x: Union[np.ndarray, torch.tensor]) -> np.ndarray:
        if self.shift_32 > 0:
            x = (x.astype(np.int32) << self.shift_32) >> self.shift_32
        else:
            x = x.astype(np.int32)
        return x / (1 << self.exp)


class FIFO(AbstractSeqModule):
    """FIFO buffer implementation. It can buffer any python object.

    Parameters
    ----------
    depth : int, optional
        Depth of the buffer, by default 1.
    """
    def __init__(self, depth: int = 1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("FIFO buffer depth needs to be more than 1.")
        self.buffer = deque([None] * depth)

    def __call__(self, x: Any) -> Any:
        return super().__call__(x)

    def forward(self, x: Any) -> Any:
        self.buffer.append(x)
        return self.buffer.popleft()
