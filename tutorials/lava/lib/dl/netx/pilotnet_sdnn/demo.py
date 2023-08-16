# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
#
# PilotNet SDNN Demo Application

import matplotlib.pyplot as plt
import numpy as np
import os
import time

from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.encoder import Compression
from lava.proc.io.sink import RingBuffer as SinkBuffer
from lava.proc.io.source import RingBuffer as SourceBuffer
from lava.proc.io.dataloader import SpikeDataloader
from lava.proc.plot.streaming import Figure, Raster, ImageView, LinePlot

from lava.lib.dl.netx.hdf5 import Network

from dataset import PilotNetDataset
from utils import (
    PilotNetEncoder, PilotNetDecoder, PilotNetMonitor,
    CustomHwRunConfig, CustomSimRunConfig,
    get_input_transform
)

from lava.utils import loihi


def use_loihi_if_installed():
    if loihi.is_installed():
        loihi.use_slurm_host(partition='oheogulch')
        print(f'Running on {loihi.host}')
    else:
        print('Lava-Loihi is not installed. Running on CPU.')


def load_network() -> Network:
    model_path = os.path.join(os.path.dirname(__file__), 'network.net')
    net = Network(net_config=model_path, skip_layers=1)
    print()
    print(net)
    print()
    return net


if __name__ == '__main__':
    use_loihi_if_installed()
    net = load_network()

    num_samples = 20000
    steps_per_sample = 1
    num_steps = num_samples + len(net.layers)
    out_offset = len(net.layers) + 3

    transform = get_input_transform(net.net_config)

    data_path = os.path.join(os.path.dirname(__file__), '../data')
    full_set = PilotNetDataset(
        path=data_path,
        size=net.inp.shape[:2],
        transform=transform,
        visualize=True,
        sample_offset=10550,
    )
    dataloader = SpikeDataloader(dataset=full_set)
    compression = Compression.DELTA_SPARSE_8 if loihi.host else Compression.DENSE
    input_encoder = PilotNetEncoder(shape=net.inp.shape,
                                    net_config=net.net_config,
                                    compression=compression)
    output_decoder = PilotNetDecoder(shape=net.out.shape)

    gt_logger = SinkBuffer(shape=(1,), buffer=num_steps)
    output_logger = SinkBuffer(shape=net.out_layer.shape, buffer=num_steps)

    dataloader.ground_truth.connect(gt_logger.a_in)
    dataloader.s_out.connect(input_encoder.inp)

    input_encoder.out.connect(net.inp)
    net.out.connect(output_decoder.inp)
    output_decoder.out.connect(output_logger.a_in)

    image = ImageView(shape=dataloader.s_out.shape,
                      bias=transform['bias'], range=transform['weight'],
                      transpose=[1, 0, 2], subplot=121)
    dataloader.s_out.connect(image.img_in)

    raster = Raster(shape=net.layers[-2].out.shape, subplot=222)
    net.layers[-2].out.connect(raster.spk_in)

    lines = LinePlot(length=1000, min=-np.pi, max=np.pi, num_lines=2,
                     subplot=224)
    output_decoder.out.connect(lines.y_in[0])
    dataloader.ground_truth.connect(lines.y_in[1])

    figure = Figure(plots=[image, raster, lines])

    print('Compiling network...')
    run_config = CustomHwRunConfig() if loihi.host else CustomSimRunConfig()
    start = time.perf_counter()
    net.create_runtime(run_cfg=run_config)
    elapsed = time.perf_counter() - start
    print(f'Done ({elapsed:.1f}s).')

    print('Running network...')
    net.run(condition=RunSteps(num_steps=num_steps, blocking=False))
    figure.show()
    print('Done.')
    if net.runtime._is_running:
        net.stop()
