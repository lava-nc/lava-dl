# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import Dict, Tuple
import numpy as np
from PIL import Image
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg

from lava.proc import io
from lava.proc.conv.process import Conv
from lava.proc.io.encoder import Compression
from lava.lib.dl.netx.utils import NetDict

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
    from lava.proc import embedded_io as eio
    from lava.magma.core.model.c.ports import CRefPort
    from lava.magma.core.model.c.type import LavaCType, LavaCDataType


class Buffer(AbstractProcess):
    """A simple buffer process that introduces a time step delay.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the buffer.
    """

    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=Buffer, protocol=LoihiProtocol)
@requires(CPU)
class PyBufferModel(PyLoihiProcessModel):
    """Buffer model."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.data = np.zeros(proc_params['shape'])

    def run_spk(self):
        self.out.send(self.data)
        self.data = self.inp.recv()


# Input Encoder ###############################################################
class PilotNetEncoder(AbstractProcess):
    """Input encoder process for PilotNet LIF.

    Parameters
    ----------
    shape : tuple of ints
        Shape of input.
    compression : Compression mode enum
        Compression mode of encoded data.
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 interval: int = 1,
                 offset: int = 0,
                 compression: Compression = Compression.DELTA_SPARSE_8) -> None:
        super().__init__(shape=shape,
                         interval=interval,
                         offset=offset % interval,
                         compression=compression)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=PilotNetEncoder, protocol=LoihiProtocol)
@requires(CPU)
class PilotNetPyEncoderModel(AbstractSubProcessModel):
    """PilotNet encoder model for CPU."""

    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        interval = proc.proc_params.get('interval')
        offset = proc.proc_params.get('offset')
        compression = proc.proc_params.get('compression')
        self.encoder = io.encoder.DeltaEncoder(shape=shape,
                                               vth=0,
                                               compression=compression)
        self.resetter = io.reset.Reset(interval=interval, offset=offset)
        self.resetter.connect_var(self.encoder.act)
        self.buffer = Buffer(shape=shape)
        proc.inp.connect(self.encoder.a_in)
        self.encoder.s_out.connect(self.buffer.inp)
        self.buffer.out.connect(proc.out)


@implements(proc=PilotNetEncoder, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PilotNetNxEncoderModel(AbstractSubProcessModel):
    """PilotNet encoder model for Loihi 2."""

    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        interval = proc.proc_params.get('interval')
        offset = proc.proc_params.get('offset')
        compression = proc.proc_params.get('compression')
        self.encoder = io.encoder.DeltaEncoder(shape=shape,
                                               vth=0,
                                               compression=compression)
        self.resetter = io.reset.Reset(interval=interval, offset=offset)
        self.resetter.connect_var(self.encoder.act)
        self.adapter = eio.spike.PyToN3ConvAdapter(shape=shape,
                                                   num_message_bits=16,
                                                   compression=compression)
        self.conv = Conv(weight=np.eye(3).reshape(3, 1, 1, 3),
                         input_shape=shape, num_message_bits=16)

        proc.inp.connect(self.encoder.a_in)
        self.encoder.s_out.connect(self.adapter.inp)
        self.adapter.out.connect(self.conv.s_in)
        self.conv.a_out.connect(proc.out)

# Voltage Reader ##############################################################


class VoltageReader(AbstractProcess):
    def __init__(self,
                 shape: Tuple[int, ...],
                 interval: int = 16,
                 offset: int = 0) -> None:
        super().__init__(shape=shape,
                         interval=interval,
                         offset=offset % interval)
        self.ref = RefPort(shape=shape)
        self.out = OutPort(shape=shape)

    def connect_var(self, var: Var) -> None:
        """Attach a var to read the state from.

        Parameters
        ----------
        var : Var
            The variable whose state needs to be read.
        """
        self.ref.connect_var(var)


@implements(proc=VoltageReader, protocol=LoihiProtocol)
@requires(CPU)
class PyVoltageReaderModel(PyLoihiProcessModel):
    """Voltage reader model for CPU."""
    ref: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.interval = proc_params['interval']
        self.offset = proc_params['offset']
        self.data = np.zeros(proc_params['shape'])

    def run_spk(self):
        if (self.time_step - 1) % self.interval == (self.offset + 1) % self.interval:
            self.out.send(self.data)

    def post_guard(self) -> None:
        return (self.time_step - 1) % self.interval == self.offset

    def run_post_mgmt(self) -> None:
        self.data = self.ref.read()

# Output Decoder ##############################################################


class PilotNetDecoder(AbstractProcess):
    """A simple affine transformer process that transforms the input.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the buffer.
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 weight: float = 1,
                 bias: float = 0,
                 interval: int = 1,
                 offset: int = 0) -> None:
        super().__init__(shape=shape,
                         weight=weight, bias=bias,
                         interval=interval, offset=offset)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=PilotNetDecoder, protocol=LoihiProtocol)
@requires(CPU)
class PyPilotNetDecoderModel(PyLoihiProcessModel):
    """Affine transformer model."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.weight = proc_params['weight']
        self.bias = proc_params['bias']
        self.interval = proc_params['interval']
        self.offset = proc_params['offset']
        self.data = np.zeros(proc_params['shape'])

    def run_spk(self):
        if (self.time_step - 1) % self.interval == self.offset:
            data = self.inp.recv()
            # reinterpret the data as 24 bit
            data = (data.astype(np.int32) << 8) >> 8
            self.data = data * self.weight + self.bias
        self.out.send(self.data)


# Monitor #####################################################################
class PilotNetMonitor(AbstractProcess):
    def __init__(self,
                 shape: Tuple[int, ...],
                 transform: Dict[str, float],
                 interval: int = 1,
                 output_offset=0) -> None:
        """PilotNet monitor process.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of input.
        transform : Dict[str, float]
            Input transformation specs.
        output_offset : int, optional
            Latency of output, by default 0.
        """
        super().__init__(shape=shape, transform=transform, interval=interval)
        self.frame_in = InPort(shape=shape)
        self.output_in = InPort(shape=(1,))
        self.gt_in = InPort(shape=(1,))
        self.proc_params['output_offset'] = output_offset


@implements(proc=PilotNetMonitor, protocol=LoihiProtocol)
@requires(CPU)
class PilotNetMonitorModel(PyLoihiProcessModel):
    """PilotNet monitor model."""
    frame_in = LavaPyType(PyInPort.VEC_DENSE, float)
    output_in = LavaPyType(PyInPort.VEC_DENSE, float)
    gt_in = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params=None) -> None:
        super().__init__(proc_params=proc_params)
        self.fig = plt.figure(figsize=(15, 5))
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        output_offset = self.proc_params['output_offset']
        interval = self.proc_params['interval']
        transform = self.proc_params['transform']
        self.weight = transform['weight']
        self.bias = transform['bias']
        self.interval = interval
        self.output_offset = output_offset
        self.gt_history = [0] * output_offset
        self.steering = Image.open('images/pilotnet_steering.png')
        self.output_history = []

    def run_spk(self) -> None:
        frame_data = self.frame_in.recv()
        output_data = self.output_in.recv()[0]
        gt_data = self.gt_in.recv()[0]
        if self.time_step < (self.output_offset + 1):
            angle = 0
        else:
            angle = output_data

        if (self.time_step - 1) % self.interval == 1:
            self.gt_history.append(gt_data)
            self.output_history.append(angle)
            frame = (frame_data.transpose([1, 0, 2])
                     - self.bias) / (2 * self.weight) + 0.5

            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax1.imshow(frame)
            self.ax2.imshow(self.steering.rotate(-angle * 180 / np.pi,
                                                 fillcolor=(255, 255, 255)))
            print(f'ground_truth = {self.gt_history[-1]}, '
                  f'prediction = {angle}')
            self.ax3.plot(np.array(self.gt_history), label='ground truth')
            self.ax3.plot(np.array(self.output_history),
                          label='network prediction')
            self.ax1.set_title('Input Frame')
            self.ax2.set_title('Network Prediction')
            self.ax3.set_ylabel('Prediction angle')
            self.ax3.set_xlabel('Frames')
            self.ax3.legend()
            clear_output(wait=True)
            display(self.fig)


loihi2hw_exception_map = {
    io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelSparse,
    PilotNetEncoder: PilotNetNxEncoderModel,
    # PilotNetDecoder: PilotNetNxDecoderModel,
}


loihi2sim_exception_map = {
    io.sink.RingBuffer: io.sink.PyReceiveModelFloat,
    io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelDense,
    PilotNetEncoder: PilotNetPyEncoderModel,
    # PilotNetDecoder: PilotNetPyDecoderModel,
}
