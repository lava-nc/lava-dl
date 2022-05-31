# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import subprocess
import os
import unittest
import typing as ty
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.magma.core.model.model import AbstractProcessModel

from lava.lib.dl import netx
from .dataset_sdnn import PilotNetDataset


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


class TestPilotNetSdnn(unittest.TestCase):
    run_it_tests: int = int(os.environ.get("RUN_IT_TESTS",
                                           0))

    @unittest.skipUnless(run_it_tests == 1,
                         "")
    def test_pilotnet_sdnn(self):
        repo_dir = subprocess.Popen(
            ['git', 'rev-parse', '--show-toplevel'],
            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
        pilotnet_sdnn_path = repo_dir + \
            "/tutorials" \
            "/lava/lib/dl/netx/pilotnet_sdnn"
        dataset_path: str = os.environ.get("PILOTNET_DATASET_PATH",
                                           "../data")
        net = netx.hdf5.Network(net_config=(pilotnet_sdnn_path
                                            + "/network.net"))

        print(net)

        print(f'There are {len(net)} layers in network:')

        for l in net.layers:
            print(f'{l.__class__.__name__:5s} \
                  : {l.name:10s}, shape : {l.shape}')

        num_samples = 200
        num_steps = num_samples + len(net.layers)

        full_set = PilotNetDataset(
            path=dataset_path,
            size=[100, 33],
            transform=net.in_layer.transform,  # input transform
            visualize=True,  # visualize ensures images are returned in sequence
            sample_offset=10550,
        )
        # train_set = PilotNetDataset(
        #     path=dataset_path,
        #     size=[100, 33],
        #     transform=net.in_layer.transform, # input transform
        #     train=True,
        # )
        # test_set = PilotNetDataset(
        #     path=dataset_path,
        #     size=[100, 33],
        #     transform=net.in_layer.transform, # input transform
        #     train=False,
        # )

        dataloader = io.dataloader.SpikeDataloader(dataset=full_set)

        gt_logger = io.sink.RingBuffer(shape=(1,), buffer=num_steps)
        output_logger = io.sink.RingBuffer(shape=net.out_layer.shape,
                                           buffer=num_steps)
        dataloader.ground_truth.connect(gt_logger.a_in)
        dataloader.s_out.connect(net.in_layer.neuron.a_in)
        net.out_layer.out.connect(output_logger.a_in)

        run_config = CustomRunConfig(select_tag='fixed_pt')
        net.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
        output = output_logger.data.get().flatten()
        gts = gt_logger.data.get().flatten()
        net.stop()

        plt.figure(figsize=(15, 10))
        plt.plot(np.array(gts[1:]), label='Ground Truth')
        plt.plot(np.array(output[len(net.layers):]).flatten()/(1 << 18),
                 label='Lava output')
        plt.xlabel('Sample frames (+10550)')
        plt.ylabel('Steering angle (radians)')
        plt.legend()