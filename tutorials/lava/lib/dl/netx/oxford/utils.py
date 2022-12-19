# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from bz2 import compress
from typing import Dict, Tuple
import numpy as np
from PIL import Image
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg

from lava.lib.dl import slayer

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
    from lava.proc import embedded_io as eio


class CustomHwRunConfig(Loihi2HwCfg):
    """Custom Loihi2 hardware run config."""
    def __init__(self):
        super().__init__(select_sub_proc_model=True)

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, InputAdapter):
            return NxInputAdapterModel
        if isinstance(proc, OutputAdapter):
            return NxOutputAdapterModel
        return super().select(proc, proc_models)


class CustomSimRunConfig(Loihi2SimCfg):
    """Custom Loihi2 simulation run config."""
    def __init__(self):
        super().__init__(select_tag='fixed_pt')

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, InputAdapter):
            return PyInputAdapterModel
        if isinstance(proc, OutputAdapter):
            return PyOutputAdapterModel
        return super().select(proc, proc_models)


# Input adapter #############################################################
class InputAdapter(AbstractProcess):
    """Input adapter process.

    Parameters
    ----------
    shape : tuple of ints
        Shape of input.
    """
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyInputAdapterModel(PyLoihiProcessModel):
    """Input adapter model for CPU."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        self.out.send(self.inp.recv())


@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxInputAdapterModel(AbstractSubProcessModel):
    """Input adapter model for Loihi 2."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        self.adapter = eio.spike.PyToNxAdapter(shape=shape)
        proc.inp.connect(self.adapter.inp)
        self.adapter.out.connect(proc.out)


# Output adapter #############################################################
class OutputAdapter(AbstractProcess):
    """Output adapter process.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of output.
    """
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputAdapterModel(PyLoihiProcessModel):
    """Output adapter model for CPU."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        self.out.send(self.inp.recv())


@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxOutputAdapterModel(AbstractSubProcessModel):
    """Output adapter model for Loihi 2."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        self.adapter = eio.spike.NxToPyAdapter(shape=shape)
        proc.inp.connect(self.adapter.inp)
        self.adapter.out.connect(proc.out)


# Monitor #####################################################################
class OxfordMonitor(AbstractProcess):
    def __init__(self, shape, gt_event) -> None:
        """Oxford monitor process.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of input.
        output_offset : int, optional
            Latency of output, by default 0.
        """
        super().__init__(shape=shape, gt_event=gt_event)
        self.spike_in = InPort(shape=shape)


@implements(proc=OxfordMonitor, protocol=LoihiProtocol)
@requires(CPU)
class OxfordMonitorModel(PyLoihiProcessModel):
    """Oxford monitor model."""
    spike_in = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params=None) -> None:
        super().__init__(proc_params=proc_params)
        self.fig = plt.figure(figsize=(7, 7))
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.gt_event = self.proc_params['gt_event']
        self.out_event = slayer.io.Event(x_event=np.array([0]),
                                         y_event=None,
                                         c_event=np.array([0]),
                                         t_event=np.array([-1]))

    def run_spk(self) -> None:
        spike_data = self.spike_in.recv()
        event = np.argwhere(spike_data != 0)
        self.out_event.x = np.concatenate([self.out_event.x,
                                           event[:, 0]])
        self.out_event.c = np.concatenate([self.out_event.c,
                                           0 * event[:, 0]])
        self.out_event.t = np.concatenate([self.out_event.t,
                                           self.time_step - 1 + 0 * event[:, 0]])

        self.ax1.clear()
        print(f'{self.time_step=}')
        self.ax1.plot(self.gt_event.t, self.gt_event.x,
                      '.', markersize=6, label='Ground Truth')
        self.ax1.plot(self.out_event.t, self.out_event.x,
                      '.', markersize=2, label='Lava')
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('Neuron ID')
        self.ax1.legend()
        clear_output(wait=True)
        display(self.fig)
