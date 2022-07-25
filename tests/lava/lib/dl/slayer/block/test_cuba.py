# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest
import h5py

import numpy as np
import torch

from lava.lib.dl import slayer, netx
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.conv import utils
from lava.proc import io

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False
# Enabling torch sometimes causes multiprocessing error, especially in unittests
utils.TORCH_IS_AVAILABLE = False

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

neuron_param = {'threshold': 0.5,
                'current_decay': 0.5,
                'voltage_decay': 0.5}


class TestCUBA(unittest.TestCase):
    """Test CUBA blocks"""

    def test_dense_block(self):
        """Test dense block with lava process implementation."""
        in_features = 10
        out_features = 5
        time_steps = 10

        # create slayer network and evaluate output
        net = slayer.block.cuba.Dense(neuron_param, in_features, out_features)
        x = (torch.rand([1, in_features, time_steps]) > 0.5).float()
        y = net(x)

        # export slayer network
        net.export_hdf5(h5py.File(tempdir + '/cuba_dense.net',
                                  'w').create_group('layer/0'))

        # create equivalent lava network using netx and evaluate output
        lava_net = netx.hdf5.Network(net_config=tempdir + '/cuba_dense.net')
        source = io.source.RingBuffer(data=x[0])
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
            print(y[0])
            print('lava output:')
            print(output)

        self.assertTrue(np.abs(y[0].data.numpy() - output).sum() == 0)

    def test_conv_block(self):
        """Test conv block with lava process implementation."""
        height = 16
        width = 24
        in_features = 3
        out_features = 5
        kernel_size = 3
        time_steps = 10

        # create slayer network and evaluate output
        net = slayer.block.cuba.Conv(neuron_param,
                                     in_features, out_features, kernel_size)
        x = (torch.rand([1, in_features,
                         height, width, time_steps]) > 0.5).float()
        y = net(x).permute((0, 3, 2, 1, 4))

        # export slayer network
        net.export_hdf5(h5py.File(tempdir + '/cuba_conv.net',
                                  'w').create_group('layer/0'))

        # create equivalent lava network using netx and evaluate output
        lava_net = netx.hdf5.Network(net_config=tempdir + '/cuba_conv.net',
                                     input_shape=(width, height, in_features))
        source = io.source.RingBuffer(data=x[0].permute((2, 1, 0, 3)))
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
            print(y[0][0, 0, 0])
            print('lava output:')
            print(output[0, 0, 0])

        self.assertTrue(np.abs(y[0].data.numpy() - output).sum() == 0)
