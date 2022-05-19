# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest
import h5py

import numpy as np
import torch
import torch.nn.functional as F

from lava.lib.dl import slayer, netx
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False

# seed = np.random.randint(1000)
seed = 196
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if verbose:
    print(f'{seed=}')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    if verbose:
        print('CUDA is not available in the system. '
              'Testing for CPU version only.')
    device = torch.device('cpu')

tempdir = os.path.dirname(__file__) + '/temp'
os.makedirs(tempdir, exist_ok=True)

neuron_param = {'threshold': 0.1,
                'activation': F.relu}


class TestCUBA(unittest.TestCase):
    """Test CUBA blocks"""

    @unittest.skip
    def test_dense_block(self):
        """Test dense block with lava process implementation."""
        in_features = 10
        out_features = 5
        time_steps = 10

        # create slayer network and evaluate output
        net = slayer.block.sigma_delta.Dense(
            neuron_param, in_features, out_features)
        x = slayer.utils.quantize(torch.rand([1, in_features, time_steps]),
                                  step=1 / net.neuron.w_scale)
        y = net(x)
        y = slayer.axon.delay(y)

        # export slayer network
        net.export_hdf5(h5py.File(tempdir + '/sdn_dense.net',
                                  'w').create_group('layer/0'))

        # create equivalent lava network using netx and evaluate output
        lava_net = netx.hdf5.Network(net_config=tempdir + '/sdn_dense.net',
                                     has_graded_input=True)
        source = io.source.RingBuffer(x[0] * net.neuron.s_scale)
        sink = io.sink.RingBuffer(shape=lava_net.out.shape, buffer=time_steps)
        source.s_out.connect(lava_net.inp)
        lava_net.out.connect(sink.a_in)
        run_condition = RunSteps(num_steps=time_steps)
        run_config = Loihi1SimCfg(select_tag='fixed_pt')
        lava_net.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        lava_net.stop()

        if verbose:
            print()
            print(lava_net)
            print('slayer output:')
            print(y[0] * net.neuron.s_scale)
            print('lava output:')
            print(output)

        self.assertTrue(np.abs(y[0].data.numpy() * net.neuron.s_scale
                               - output).sum() == 0)
