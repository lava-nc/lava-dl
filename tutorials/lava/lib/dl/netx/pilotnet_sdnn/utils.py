# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from bz2 import compress
from typing import Dict, Tuple
import numpy as np
from PIL import Image
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from lava.magma.core.process.process import AbstractProcess
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

from lava.proc import io
from lava.proc.io.encoder import Compression
from lava.lib.dl.netx.utils import NetDict

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
    from lava.proc import embedded_io as eio


class CustomHwRunConfig(Loihi2HwCfg):
    """Custom Loihi2 hardware run config."""
    def __init__(self):
        super().__init__(select_sub_proc_model=True)

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, io.encoder.DeltaEncoder):
            return io.encoder.PyDeltaEncoderModelSparse
        if isinstance(proc, PilotNetEncoder):
            return PilotNetNxEncoderModel
        if isinstance(proc, PilotNetDecoder):
            return PilotNetNxDecoderModel
        return super().select(proc, proc_models)


class CustomSimRunConfig(Loihi2SimCfg):
    """Custom Loihi2 simulation run config."""
    def __init__(self):
        super().__init__(select_tag='fixed_pt')

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, io.sink.RingBuffer):
            return io.sink.PyReceiveModelFloat
        if isinstance(proc, io.encoder.DeltaEncoder):
            return io.encoder.PyDeltaEncoderModelDense
        if isinstance(proc, PilotNetEncoder):
            return PilotNetPyEncoderModel
        if isinstance(proc, PilotNetDecoder):
            return PilotNetPyDecoderModel
        return super().select(proc, proc_models)


def get_input_transform(net_config: NetDict) -> Dict[str, float]:
    """Extract input transformation from the network configuration

    Parameters
    ----------
    net_config : NetDict
        Network configuration.

    Returns
    -------
    Dict[str, float]
        Input transformation descriptor.
    """
    transform = {'weight': 64, 'bias': 0}
    if 'weight' in net_config['layer'][0].keys():
        transform['weight'] = net_config['layer'][0]['weight']
    if 'bias' in net_config['layer'][0].keys():
        transform['bias'] = net_config['layer'][0]['bias']
    return transform


# Input Encoder ###############################################################
class PilotNetEncoder(AbstractProcess):
    """Input encoder process for PilotNet SDNN.

    Parameters
    ----------
    shape : tuple of ints
        Shape of input.
    net_config : NetDict
        Network configuration.
    compression : Compression mode enum
        Compression mode of encoded data.
    """
    def __init__(self,
                 shape: Tuple[int, ...],
                 net_config: NetDict,
                 compression: Compression = Compression.DELTA_SPARSE_8) -> None:
        super().__init__(shape=shape,
                         vth=net_config['layer'][0]['neuron']['vThMant'],
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
        vth = proc.proc_params.get('vth')
        self.encoder = io.encoder.DeltaEncoder(shape=shape,
                                               vth=vth,
                                               spike_exp=6)
        proc.inp.connect(self.encoder.a_in)
        self.encoder.s_out.connect(proc.out)


@implements(proc=PilotNetEncoder, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PilotNetNxEncoderModel(AbstractSubProcessModel):
    """PilotNet encoder model for Loihi 2."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        vth = proc.proc_params.get('vth')
        compression = proc.proc_params.get('compression')
        self.encoder = io.encoder.DeltaEncoder(shape=shape, vth=vth,
                                               compression=compression)
        self.adapter = eio.spike.PyToN3ConvAdapter(shape=shape,
                                                   num_message_bits=16,
                                                   spike_exp=6,
                                                   compression=compression)
        proc.inp.connect(self.encoder.a_in)
        self.encoder.s_out.connect(self.adapter.inp)
        self.adapter.out.connect(proc.out)


# Output Decoder ##############################################################
class PilotNetDecoder(AbstractProcess):
    """Output decoder process for PilotNet SDNN.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of output.
    """
    def __init__(self,
                 shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=PilotNetDecoder, protocol=LoihiProtocol)
@requires(CPU)
class PilotNetPyDecoderModel(PyLoihiProcessModel):
    """PilotNet decoder model for CPU."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        raw_data = self.inp.recv()
        data = raw_data / (1 << 18)
        self.out.send(data)


class PilotNetFixedPtDecoder(AbstractProcess):
    """PilotNet fixed point output decoder."""
    def __init__(self,
                 shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=PilotNetFixedPtDecoder, protocol=LoihiProtocol)
@requires(CPU)
class PilotNetFixedPtDecoderModel(PyLoihiProcessModel):
    """PilotNet decoder model for fixed point output."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        raw_data = self.inp.recv()
        # interpret it as a 24 bit signed integer
        raw_data = (raw_data.astype(np.int32) << 8) >> 8
        data = raw_data / (1 << 18)
        self.out.send(data)

@implements(proc=PilotNetDecoder, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PilotNetNxDecoderModel(AbstractSubProcessModel):
    """PilotNet decoder model for Loihi."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        self.decoder = PilotNetFixedPtDecoder(shape=shape)
        self.adapter = eio.spike.NxToPyAdapter(shape=shape,
                                               num_message_bits=32)
        proc.inp.connect(self.adapter.inp)
        self.adapter.out.connect(self.decoder.inp)
        self.decoder.out.connect(proc.out)


# Monitor #####################################################################
class PilotNetMonitor(AbstractProcess):
    def __init__(self,
                 shape: Tuple[int, ...],
                 transform: Dict[str, float],
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
        super().__init__(shape=shape, transform=transform)
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
        transform = self.proc_params['transform']
        self.weight = transform['weight']
        self.bias = transform['bias']
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
        self.gt_history.append(gt_data)
        self.output_history.append(angle)
        frame  = (frame_data.transpose([1, 0, 2])
                  - self.bias) / (2 * self.weight) + 0.5

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax1.imshow(frame)
        self.ax2.imshow(self.steering.rotate(-angle * 180 / np.pi,
                                             fillcolor=(255, 255, 255)))
        print(f'ground_truth = {self.gt_history[-self.output_offset + 1]}, '
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
