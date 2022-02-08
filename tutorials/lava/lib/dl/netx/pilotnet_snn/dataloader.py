# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import Iterable
import numpy as np
import random

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort, RefPort
from lava.magma.core.process.variable import Var
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import HostCPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps


class State(AbstractProcess):
    def __init__(
        self,
        dataset: Iterable,
        # set_var: Var,
        interval: int = 1,
        offset: int = 0,
    ) -> None:
        super().__init__(dataste=dataset, interval=interval, offset=offset)
        self.interval = Var((1,), interval)
        self.offset = Var((1,), offset % interval)
        # self.state = RefPort(set_var.shape)
        # self.state.connect_var(set_var)
        self.dataset = dataset
        self.proc_params['saved_objects'] = self.dataset


@implements(proc=State, protocol=LoihiProtocol)
@requires(HostCPU)  # to ensure this model always runs on host cpu
@tag('fixed_pt', 'floating_pt')
class PyStateModel(PyLoihiProcessModel):
    interval: np.ndarray = LavaPyType(np.ndarray, int)
    offset: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.id = 0
        self.dataset = self.proc_params['saved_objects']

    def post_guard(self) -> None:
        return (self.current_ts - 1) % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        print(f't={(self.current_ts - 1)}: {self.id = }')
        print(self.dataset)
        self.id += 1


if __name__ == '__main__':
    num_steps = 30
    dataloader = State('test', interval=6)

    run_condition = RunSteps(num_steps=num_steps)
    run_config = Loihi1SimCfg(select_tag='fixed_pt')
    dataloader.run(condition=run_condition, run_cfg=run_config)
    dataloader.stop()
