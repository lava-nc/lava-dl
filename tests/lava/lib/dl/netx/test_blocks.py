# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from typing import List
import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.io.source import RingBuffer as SendProcess
from lava.proc.io.sink import RingBuffer as ReceiveProcess
from lava.proc.lif.process import LIF

from lava.lib.dl.netx.blocks.process import Dense, Conv, Input


verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False
HAVE_DISPLAY = 'DISPLAY' in os.environ
root = os.path.dirname(os.path.abspath(__file__))


class TestRunConfig(RunConfig):
    """Run configuration selects appropriate ProcessModel based on tag:
    floating point precision or Loihi bit-accurate fixed point precision"""
    def __init__(self, select_tag: str = 'fixed_pt') -> None:
        super().__init__(custom_sync_domains=None)
        self.select_tag = select_tag

    def select(
        self, _: AbstractProcess, proc_models: List[PyLoihiProcessModel]
    ) -> PyLoihiProcessModel:
        for pm in proc_models:
            if self.select_tag in pm.tags:
                return pm
        raise AssertionError('No legal ProcessModel found.')


class TestLIFBlocks(unittest.TestCase):
    def test_input(self) -> None:
        num_steps = 15
        lif_params = {
            'vth': 25,
            'du': 4095,
            'dv': 1024,
            'bias_exp': 6,
            'graded_spikes': False,
        }

        input_blk = Input(
            shape=(16, 16, 3),
            neuron_params={'neuron_proc': LIF, **lif_params},
            bias=np.load(root + '/gts/tinynet/input_bias.npy')
        )
        sink = ReceiveProcess(shape=input_blk.out.shape, buffer=num_steps)
        input_blk.out.connect(sink.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        input_blk.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        input_blk.stop()

        gt = np.load(root + '/gts/tinynet/input.npy')[0].transpose([2, 1, 0, 3])
        gt = gt[..., 1:]  # there is an offset of one time step

        error = np.abs(output - gt).sum()
        if verbose:
            print('Input spike error:', error)
            if HAVE_DISPLAY:
                plt.figure()
                out_ae = np.argwhere(output.reshape((-1, num_steps)) > 0)
                gt_ae = np.argwhere(gt.reshape((-1, num_steps)) > 0)
                plt.plot(
                    gt_ae[:, 1],
                    gt_ae[:, 0],
                    '.', markersize=15, label='Ground Truth'
                )
                plt.plot(out_ae[:, 1], out_ae[:, 0], '.', label='Input Block')
                plt.xlabel('Time')
                plt.ylabel('Neuron ID')
                plt.legend()
                plt.show()

        self.assertTrue(
            error == 0,
            f'Output spike and ground truth do not match for Input block. '
            f'Found {output[output != gt] = } and {gt[output != gt] = }. '
            f'Error was {error}.'
        )

    def test_conv(self) -> None:
        num_steps = 16
        lif_params = {
            'vth': 25,
            'du': 4095,
            'dv': 1024,
            'bias_exp': 6,
            'graded_spikes': False,
        }

        conv_blk = Conv(
            shape=(6, 6, 24),
            input_shape=(16, 16, 3),
            neuron_params={'neuron_proc': LIF, **lif_params},
            weight=np.load(root + '/gts/tinynet/layer1_kernel.npy'),
            bias=np.load(root + '/gts/tinynet/layer1_bias.npy'),
            stride=2,
        )
        data = np.load(
            root + '/gts/tinynet/input.npy'
        )[0].transpose([2, 1, 0, 3])
        source = SendProcess(data=data[..., 1:])
        sink = ReceiveProcess(shape=conv_blk.out.shape, buffer=num_steps)
        source.s_out.connect(conv_blk.inp)
        conv_blk.out.connect(sink.a_in)

        run_condition = RunSteps(num_steps=1)
        run_config = TestRunConfig(select_tag='fixed_pt')
        current = []
        for t in range(num_steps):
            conv_blk.run(condition=run_condition, run_cfg=run_config)
            if t == 0:
                conv_blk.neuron.u.set(np.zeros(conv_blk.neuron.shape))
                conv_blk.neuron.v.set(np.zeros(conv_blk.neuron.shape))
            current.append(conv_blk.neuron.u.get())
        s = sink.data.get()
        conv_blk.stop()

        s[..., 0] = 0
        u = np.stack(current).transpose([1, 2, 3, 0])
        s_gt = np.load(root + '/gts/tinynet/layer1.npy')[0].transpose(
            [2, 1, 0, 3]
        )
        u_gt = np.load(root + '/gts/tinynet/current.npy')[0].transpose(
            [2, 1, 0, 3]
        )
        u_gt *= 64

        s_error = np.abs(s - s_gt).sum()
        u_error = np.abs(u - u_gt).sum()
        if verbose:
            print('Conv spike error:', s_error)
            if HAVE_DISPLAY:
                plt.figure()
                out_ae = np.argwhere(s.reshape((-1, num_steps)) > 0)
                gt_ae = np.argwhere(s_gt.reshape((-1, num_steps)) > 0)
                plt.plot(
                    gt_ae[:, 1],
                    gt_ae[:, 0],
                    '.', markersize=15, label='Ground Truth'
                )
                plt.plot(out_ae[:, 1], out_ae[:, 0], '.', label='Input Block')
                plt.xlabel('Time')
                plt.ylabel('Neuron ID')
                plt.legend()
                plt.show()

        self.assertTrue(
            s_error == 0,
            f'Output spike and ground truth do not match for Conv block. '
            f'Found {s[s != s_gt] = } and {s_gt[s != s_gt] = }. '
            f'Error was {s_error}.'
        )

        self.assertTrue(
            u_error == 0,
            f'Output current and ground truth do not match for Conv block. '
            f'Found {u[u != u_gt] = } and {u_gt[u != u_gt] = }. '
            f'Error was {u_error}.'
        )

    def test_dense(self) -> None:
        num_steps = 2000
        lif_params = {
            'vth': 6,
            'du': 4095,
            'dv': 415,
            'bias_exp': 6,
            'graded_spikes': False,
        }

        dense_blk = Dense(
            shape=(256,),
            neuron_params={'neuron_proc': LIF, **lif_params},
            weight=np.load(root + '/gts/dense/weight.npy')
        )

        source = SendProcess(data=np.load(root + '/gts/dense/in.npy'))
        sink = ReceiveProcess(shape=dense_blk.out.shape, buffer=num_steps)
        source.s_out.connect(dense_blk.inp)
        dense_blk.out.connect(sink.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        dense_blk.run(condition=run_condition, run_cfg=run_config)
        s = sink.data.get()
        dense_blk.stop()

        s_gt = np.load(root + '/gts/dense/out.npy')
        s_error = np.abs(s - s_gt).sum()

        if verbose:
            print('Conv spike error:', s_error)
            if HAVE_DISPLAY:
                plt.figure()
                out_ae = np.argwhere(s.reshape((-1, num_steps)) > 0)
                gt_ae = np.argwhere(s_gt.reshape((-1, num_steps)) > 0)
                plt.plot(
                    gt_ae[:, 1],
                    gt_ae[:, 0],
                    '.', markersize=15, label='Ground Truth'
                )
                plt.plot(out_ae[:, 1], out_ae[:, 0], '.', label='Input Block')
                plt.xlabel('Time')
                plt.ylabel('Neuron ID')
                plt.legend()
                plt.show()

        self.assertTrue(
            s_error == 0,
            f'Output spike and ground truth do not match for Conv block. '
            f'Found {s[s != s_gt] = } and {s_gt[s != s_gt] = }. '
            f'Error was {s_error}.'
        )
