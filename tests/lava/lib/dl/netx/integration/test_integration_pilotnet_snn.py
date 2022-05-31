# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import subprocess
import os
import typing as ty
import unittest
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.magma.core.model.model import AbstractProcessModel

from lava.lib.dl import netx
from .dataset_snn import PilotNetDataset


class CustomRunConfig(Loihi2HwCfg):

    def select(self,
               proc,
               proc_models: ty.List[ty.Type[AbstractProcessModel]]):
        # customize run config to always use float model for io.sink.RingBuffer
        if isinstance(proc, io.sink.RingBuffer):
            return io.sink.PyReceiveModelFloat
        else:
            if isinstance(proc_models, list):
                return super().select(proc, proc_models)
            else:
                raise AssertionError("Process models, not a list")


class TestPilotNetSnn(unittest.TestCase):
    run_it_tests: int = int(os.environ.get("RUN_IT_TESTS",
                                           0))

    @unittest.skipUnless(run_it_tests == 1,
                         "")
    def test_pilotnet_snn(self):
        repo_dir = subprocess.Popen(
            ['git', 'rev-parse', '--show-toplevel'],
            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
        pilotnet_snn_path = repo_dir + \
            "/tutorials" \
            "/lava/lib/dl/netx/pilotnet_snn"
        dataset_path: str = os.environ.get("PILOTNET_DATASET_PATH",
                                           "../data")
        net = netx.hdf5.Network(net_config=(pilotnet_snn_path + "/network.net"))

        print(net)

        print(f'There are {len(net)} layers in network:')

        for layer in net.layers:
            print(f'{layer.__class__.__name__:5s} \
            : {layer.name:10s}, shape : {layer.shape}')

        num_samples = 200
        steps_per_sample = 16
        readout_offset = (steps_per_sample - 1) + len(net.layers)
        num_steps = num_samples * steps_per_sample

        full_set = PilotNetDataset(
            path=dataset_path,
            transform=net.in_layer.transform,  # input transform
            visualize=True,  # visualize ensures images are returned in sequence
            sample_offset=10550,
        )
        # train_set = PilotNetDataset(
        #     path=dataset_path,
        #     transform=net.in_layer.transform,  # input transform
        #     train=True,
        # )
        # test_set = PilotNetDataset(
        #     path=dataset_path,
        #     transform=net.in_layer.transform,  # input transform
        #     train=False,
        # )

        dataloader = io.dataloader.StateDataloader(
            dataset=full_set,
            interval=steps_per_sample,
        )

        gt_logger = io.sink.RingBuffer(shape=(1,), buffer=num_steps)
        output_logger = io.sink.Read(
            num_samples,
            interval=steps_per_sample,
            offset=readout_offset
        )
        # reset
        for i, l in enumerate(net.layers[:-1]):
            u_resetter = io.reset.Reset(interval=steps_per_sample, offset=i)
            v_resetter = io.reset.Reset(interval=steps_per_sample, offset=i)
            u_resetter.connect_var(l.neuron.u)
            v_resetter.connect_var(l.neuron.v)

        dataloader.ground_truth.connect(gt_logger.a_in)
        dataloader.connect_var(net.in_layer.neuron.bias)
        output_logger.connect_var(net.out_layer.neuron.v)

        run_config = CustomRunConfig(select_tag='fixed_pt')
        net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
        results = output_logger.data.get().flatten()
        gts = gt_logger.data.get().flatten()[::steps_per_sample]
        net.stop()

        results = results.flatten()/steps_per_sample/32/64
        results = results[1:] - results[:-1]
        loihi = np.load(pilotnet_snn_path + "/3x3pred.npy")

        plt.figure(figsize=(15, 10))
        plt.plot(loihi, linewidth=5, label='Loihi output')
        plt.plot(results, label='Lava output')
        plt.plot(gts, label='Ground truth')
        plt.xlabel(f'Sample frames (+{full_set.sample_offset})')
        plt.ylabel('Steering angle (radians)')
        plt.legend()

        error = np.sum((loihi - results)**2)
        print(f'{error=}')
