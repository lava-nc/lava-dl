import numpy as np
from typing import Tuple

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore
from lava.utils.dataloader.mnist import MnistDataset

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
  from lava.proc import embedded_io as eio

###############################################################################
################# I N P    A D A P T E R    P R O C E S S #####################
###############################################################################

class InputAdapter(AbstractProcess):
  """
  Input Adapter Process.
  """
  def __init__(self, shape: Tuple[int, ...]):
    super().__init__(shape=shape)
    self.inp = InPort(shape=shape)
    self.out = OutPort(shape=shape)

@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyInputAdapter(PyLoihiProcessModel):
  """
  Input adapter model for CPU, i.e., when your spike input process is on CPU and
  you plan to send the input spikes to a Loihi2 Simulation running on CPU.
  """
  inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
  out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

  def run_spk(self):
    self.out.send(self.inp.recv())

@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxInputAdapter(AbstractSubProcessModel):
  """
  Input adapter model for Loihi-2, i.e., your spikes are generated on CPU and
  you plan to send them to Loihi-2 neuro-cores.
  """
  def __init__(self, proc: AbstractProcess):
    self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
    self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
    shape = proc.proc_params.get("shape")
    self.adapter = eio.spike.PyToNxAdapter(shape=shape)
    proc.inp.connect(self.adapter.inp)
    self.adapter.out.connect(proc.out)

###############################################################################
################# O U T    A D A P T E R    P R O C E S S #####################
###############################################################################

class OutputAdapter(AbstractProcess):
  """
  Output adapater process.
  """
  def __init__(self, shape: Tuple[int, ...]):
    super().__init__(shape=shape)
    self.inp = InPort(shape=shape)
    self.out = OutPort(shape=shape)

@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputAdapter(PyLoihiProcessModel):
  """
  Output adapter model for CPU, i.e., when your SNN is running on Loihi2
  Simulation on CPU, and you plan to accept the output spikes on CPU itself.
  """
  inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
  out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

  def run_spk(self):
    self.out.send(self.inp.recv())

@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxOutputAdapter(AbstractSubProcessModel):
  """
  Output adapter mode for Loihi-2, i.e., your spikes are generated on Loihi-2
  neuro-cores and you plan to send them to CPU.
  """
  def __init__(self, proc:AbstractProcess):
    self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
    self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)

    shape = proc.proc_params.get("shape")
    self.adapter = eio.spike.NxToPyAdapter(shape=shape)
    proc.inp.connect(self.adapter.inp)
    self.adapter.out.connect(proc.out)