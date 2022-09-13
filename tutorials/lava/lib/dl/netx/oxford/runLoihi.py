import os
import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.sink import RingBuffer as ReceiveProcess
from lava.proc.io.source import RingBuffer as SendProcess
from lava.proc import snip_io as sio

from lava.lib.dl import netx
from lava.lib.dl import slayer

root = os.path.abspath(os.path.dirname(__file__))

net = netx.hdf5.Network(net_config=root + '/Trained/network.net', num_layers=1)
print(net)

print(f'There are {len(net)} layers in network:')

for l in net.layers:
    print(f'{l.block:5s} : {l.name:10s}, shape : {l.shape}')

input = slayer.io.read_np_spikes(root + '/input.npy')
target = slayer.io.read_np_spikes(root + '/output.npy')
source = SendProcess(data=input.to_tensor(dim=(1, 200, 2000)).squeeze())
sink = ReceiveProcess(shape=net.out.shape, buffer=2000)
inp_adapter = sio.spike.PyToNxAdapter(shape=net.inp.shape)
out_adapter = sio.spike.NxToPyAdapter(shape=net.out.shape)

source.s_out.connect(inp_adapter.inp)
inp_adapter.out.connect(net.inp)
net.out.connect(out_adapter.inp)
out_adapter.out.connect(sink.a_in)

import os
import logging

os.environ['SLURM'] = '1'
os.environ['PARTITION'] = 'oheogulch'
os.environ['LOIHI_GEN'] = 'N3B3'

run_condition = RunSteps(num_steps=2000)
run_config = Loihi2HwCfg()
net._log_config.level = logging.INFO
net.run(condition=run_condition, run_cfg=run_config)
output = sink.data.get()
net.stop()