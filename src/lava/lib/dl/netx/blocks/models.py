# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# from .process import Conv
from lava.lib.dl.netx.blocks.process import Input, Dense, Conv


@requires(CPU)
@tag('fixed_pt')
class AbstractPyBlockModel(AbstractSubProcessModel):
    def __init__(self, proc: AbstractProcess) -> None:
        if proc.graded_input:
            self.inp: PyInPort = LavaPyType(np.ndarray, np.int32, precision=32)
        else:
            self.inp: PyInPort = LavaPyType(np.ndarray, np.int8, precision=1)

        if proc.graded_output:
            self.out: PyOutPort = LavaPyType(np.ndarray, np.int32, precision=32)
        else:
            self.out: PyOutPort = LavaPyType(np.ndarray, np.int8, precision=1)


@implements(proc=Input, protocol=LoihiProtocol)
class PyInputModel(AbstractPyBlockModel):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)


@implements(proc=Dense, protocol=LoihiProtocol)
class PyDenseModel(AbstractPyBlockModel):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)


@implements(proc=Conv, protocol=LoihiProtocol)
class PyConvModel(AbstractPyBlockModel):
    def __init__(self, proc: AbstractProcess) -> None:
        super().__init__(proc)
