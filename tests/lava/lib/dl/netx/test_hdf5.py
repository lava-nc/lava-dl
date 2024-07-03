# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_configs import RunConfig
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.proc.conv import utils
from lava.proc.sparse.process import Sparse, DelaySparse
from lava.proc.dense.process import Dense, DelayDense

from lava.lib.dl import netx
import torch
from lava.lib.dl import slayer
import h5py
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False
HAVE_DISPLAY = 'DISPLAY' in os.environ
root = os.path.dirname(os.path.abspath(__file__))
# Enabling torch sometimes causes multiprocessing error, especially in unittests
utils.TORCH_IS_AVAILABLE = False


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


class TestHdf5Netx(unittest.TestCase):
    def test_num_layers(self) -> None:
        """Tests the number of layers generated."""
        net = netx.hdf5.Network(net_config=root + '/tiny.net', num_layers=2)
        self.assertTrue(
            len(net) <= 2,
            f'Expected less than 2 blocks in network. Found {len(net) = }.'
        )

    def test_input_transform(self) -> None:
        """Tests the input tansform of known hdf5 net."""
        net = netx.hdf5.Network(net_config=root + '/tiny.net', num_layers=1)
        bias = net.in_layer.transform['bias']
        weight = net.in_layer.transform['weight']
        self.assertEqual(
            bias, 34,
            f'Expected transformation bias to be 34. Found {bias}.'
        )
        self.assertEqual(
            weight, 36,
            f'Expected transformation weight to be 36. Found {weight}.'
        )

    def test_mnist(self) -> None:
        """Tests loading of MNIST MLP."""
        net = netx.hdf5.Network(net_config=root + '/mnist.net')
        self.assertEqual(len(net), 4)
        self.assertTrue(type(net.layers[0]) == netx.blocks.process.Input)
        self.assertTrue(type(net.layers[1]) == netx.blocks.process.Dense)
        self.assertTrue(type(net.layers[2]) == netx.blocks.process.Dense)
        self.assertTrue(type(net.layers[3]) == netx.blocks.process.Dense)

    def test_tinynet(self) -> None:
        """Tests the output of three layer CNN."""
        steps_per_sample = 17
        net = netx.hdf5.Network(net_config=root + '/tiny.net')

        num_steps = steps_per_sample + len(net)
        sink = io.sink.RingBuffer(
            shape=net.out_layer.out.shape, buffer=num_steps
        )
        net.out_layer.out.connect(sink.a_in)

        # layer reset mechanism
        for i, l in enumerate(net.layers):
            u_resetter = io.reset.Reset(
                interval=steps_per_sample, offset=i - 1)
            v_resetter = io.reset.Reset(
                interval=steps_per_sample, offset=i - 1)
            u_resetter.connect_var(l.neuron.u)
            v_resetter.connect_var(l.neuron.v)

        if verbose:
            print(f'Network created from {net.filename}')
            print(net)
            print(f'{len(net) = }')

        # set input bias for the ground truth sample
        net.in_layer.neuron.bias_mant.init = np.load(
            root + '/gts/tinynet/input_bias.npy'
        )

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        net.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        net.stop()

        gt = np.load(
            root + '/gts/tinynet/output.npy'
        )[0].transpose([2, 1, 0, 3])
        gt = gt[..., 1:]

        error = np.abs(output[..., 2:17] - gt).sum()
        if verbose:
            print('Output spike error:', error)
            if HAVE_DISPLAY:
                plt.figure()
                out_ae = np.argwhere(output.squeeze() > 0)
                gt_ae = np.argwhere(gt.squeeze() > 0)
                plt.plot(
                    gt_ae[:, 1] + len(net) - 1,
                    gt_ae[:, 0],
                    '.', markersize=15, label='Ground Truth'
                )
                plt.plot(out_ae[:, 1], out_ae[:, 0], '.', label='Output')
                plt.xlabel('Time')
                plt.ylabel('Neuron ID')
                plt.legend()
                plt.show()

        self.assertTrue(
            error == 0,
            f'Output spike and ground truth do not match. '
            f'Found {output[..., 2:17] = } and {gt = }. '
            f'Error was {error}.'
        )

    def test_pilotnet_sdnn(self) -> None:
        """Tests the output of pilotnet sdnn."""
        net_config = root + '/gts/pilotnet_sdnn/network.net'
        net = netx.hdf5.Network(net_config=net_config)
        input = np.load(root + '/gts/pilotnet_sdnn/input.npy')
        source = io.source.RingBuffer(data=input)
        sink = io.sink.RingBuffer(shape=net.out_layer.shape,
                                  buffer=len(net.layers))
        source.s_out.connect(net.in_layer.neuron.a_in)
        net.out_layer.out.connect(sink.a_in)

        num_steps = len(net.layers)
        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        net.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        net.stop()

        gt = np.load(root + '/gts/pilotnet_sdnn/output.npy')
        error = np.abs(output - gt).sum()
        if verbose:
            print('Network:')
            print(net)
            print(f'{output=}')
            print('PilotNet SDNN spike error:', error)

        self.assertTrue(
            error == 0,
            f'Output spike and ground truth do not match for PilotNet SDNN. '
            f'Found {output[output != gt] = } and {gt[output != gt] = }. '
            f'Error was {error}.'
        )

    def test_pilotnet_sdnn_spike_exp(self) -> None:
        """Tests the output of pilotnet sdnn with spike exp."""
        net_config = root + '/gts/pilotnet_sdnn/network.net'
        net = netx.hdf5.Network(net_config=net_config, spike_exp=0)
        input = np.load(root + '/gts/pilotnet_sdnn/input.npy')
        source = io.source.RingBuffer(data=input)
        sink = io.sink.RingBuffer(shape=net.out_layer.shape,
                                  buffer=len(net.layers))
        source.s_out.connect(net.in_layer.neuron.a_in)
        net.out_layer.out.connect(sink.a_in)

        num_steps = len(net.layers)
        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        net.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        net.stop()

        scale = (1 << (6 - net.spike_exp))
        gt = np.load(root + '/gts/pilotnet_sdnn/output.npy') / scale
        error = np.abs(output - gt).mean()
        if verbose:
            print('Network:')
            print(net)
            print(f'{output=}')
            print('PilotNet SDNN spike error:', error)

        self.assertTrue(
            error < 2 * scale,
            f'Output spike and ground truth do not match for PilotNet SDNN. '
            f'Found {output[output != gt] = } and {gt[output != gt] = }. '
            f'Error was {error}.'
        )

    def test_sparse_pilotnet_sdnn(self) -> None:
        """Tests sparse_fc_layer Network arg on Dense blocks"""
        net_config = root + '/gts/pilotnet_sdnn/network.net'
        net = netx.hdf5.Network(net_config=net_config, sparse_fc_layer=True)
        dense_layers = [layer for layer in net.layers
                        if isinstance(layer, netx.blocks.process.Dense)]

        self.assertTrue(
            np.all([
                isinstance(layer.synapse, Sparse) for layer in dense_layers
            ])
        )

    def test_axonal_delay_ntidigits(self) -> None:
        """Tests the output of ntidigits hdf5 description. This network
        consists of axonal delay. So this tests specifically tests for
        correctness of axonal delay."""
        net_config = root + '/gts/ntidigits/ntidigits.net'
        input = np.load(root + '/gts/ntidigits/input.npy')
        gt = np.load(root + '/gts/ntidigits/output.npy')
        num_steps = input.shape[1]

        # skipping the last average layer which is not suppprted
        net = netx.hdf5.Network(net_config=net_config, num_layers=5)

        inp_gen = io.source.RingBuffer(data=input)
        output_logger = io.sink.RingBuffer(shape=net.out_layer.shape,
                                           buffer=num_steps)

        inp_gen.s_out.connect(net.inp)
        net.out.connect(output_logger.a_in)

        run_condition = RunSteps(num_steps=num_steps)
        run_config = TestRunConfig(select_tag='fixed_pt')
        net.run(condition=run_condition, run_cfg=run_config)
        output = output_logger.data.get()
        net.stop()

        out_ev = np.argwhere(output > 0)
        gt_ev = np.argwhere(gt > 0)

        error = np.abs(output[:, -1] - gt[:, 1]).sum()

        if verbose:
            if bool(os.environ.get('DISPLAY', None)):
                plt.figure(figsize=(10, 5))
                plt.plot(out_ev[:, 1], out_ev[:, 0], '.',
                         markersize=12, label='Output Spikes')
                plt.plot(gt_ev[:, 1], gt_ev[:, 0], '.', label='GT Spikes')
                plt.xlabel(f'time')
                plt.ylabel('Neuron ID')
                plt.legend()
                plt.show()

        self.assertTrue(
            error == 0,
            f'Output spike and ground truth do not match for NTIDIGITS network.'
            f'Found {output[output != gt] = } and {gt[output != gt] = }. '
            f'Error was {error}.'
        )

    def test_sparse_axonal_delay_ntidigits(self) -> None:
        """Tests that sparse axonal delays work on Dense Blocks."""
        net_config = root + '/gts/ntidigits/ntidigits.net'
        # skipping the last average layer which is not suppprted
        net = netx.hdf5.Network(net_config=net_config, num_layers=5,
                                sparse_fc_layer=True)
        dense_layers = [layer for layer in net.layers
                        if isinstance(layer, netx.blocks.process.Dense)]

        self.assertTrue(
            np.all([
                isinstance(layer.synapse, (Sparse, DelaySparse))
                for layer in dense_layers
            ])
        )

    def test_recurrent_netx_save_and_load(self) -> None:
        """Tests that recurrent network can be saved and loaded.
        Verifies that an toy recurrent network runs in Lava and matches
        Lava-dl output firing rates within 2% error margin."""

        class Network(torch.nn.Module):
            def __init__(
                self,
            ):
                super(Network, self).__init__()

                self.cuba_params = {
                    "threshold": 0.95,
                    "current_decay": 0.5,
                    "voltage_decay": 0.15,
                    "tau_grad": 1.0,
                    "scale_grad": 1.0,
                    "shared_param": False,
                    "requires_grad": False,
                    "graded_spike": False,
                }
                self.blocks = torch.nn.ModuleList(
                    [
                        slayer.block.cuba.Recurrent(
                            self.cuba_params,
                            6,
                            5,
                            weight_scale=0.0,
                            pre_hook_fx=None,
                        ),
                        slayer.block.cuba.Recurrent(
                            self.cuba_params,
                            5,
                            5,
                            weight_scale=0.0,
                            pre_hook_fx=None,
                        ),
                    ]
                )

                self.blocks[0].input_synapse.weight.data += 0.1
                self.blocks[0].recurrent_synapse.weight.data += 0.05

                self.blocks[1].input_synapse.weight.data += 0.1
                self.blocks[1].recurrent_synapse.weight.data += 0.05

            def forward(self, spike):
                for block in self.blocks:
                    spike = block(spike)
                return spike

            def export_hdf5(self, filename):
                # network export to hdf5 format
                with h5py.File(filename, "w") as h:
                    layer = h.create_group("layer")
                    for i, b in enumerate(self.blocks):
                        b.export_hdf5(layer.create_group(f"{i}"))

        current_file_directory = os.path.dirname(os.path.abspath(__file__))

        # batch = 1, channels = 6, time = 101
        num_steps = 101
        input = torch.zeros((1, 6, num_steps))
        input += 0.2

        net = Network()
        lava_dl_output = net(input).detach().numpy()

        filename = os.path.join(current_file_directory, "recurrent_example.net")
        net.export_hdf5(filename)

        net_lava = netx.hdf5.Network(filename, input_message_bits=24)

        net_lava_input_scale_factor_exp = 10
        net_lava_input_scale_factor = 2**net_lava_input_scale_factor_exp
        net_lava.layers[0].synapse_rec.proc_params._parameters[
            "weight_exp"
        ] += net_lava_input_scale_factor_exp
        net_lava.layers[0].neuron.vth.init *= net_lava_input_scale_factor
        net_lava.layers[0].neuron.bias_mant.init *= net_lava_input_scale_factor

        input_lava = input[0].numpy() * net_lava_input_scale_factor

        source = io.source.RingBuffer(data=input_lava)
        sink = io.sink.RingBuffer(
            shape=net_lava.out.shape, buffer=num_steps + 1
        )
        source.s_out.connect(net_lava.inp)
        net_lava.out.connect(sink.a_in)

        run_config = Loihi2SimCfg(select_tag="fixed_pt")
        run_condition = RunSteps(num_steps=num_steps)
        net_lava.run(condition=run_condition, run_cfg=run_config)
        lava_output = sink.data.get()
        net_lava.stop()

        eps = 0.02
        assert (
            np.abs(
                (lava_dl_output[0][0].mean() - lava_output[0].mean())
                / lava_dl_output[0][0].mean()
            )
            <= eps
        ), f"""Mean firing rate mismatch {lava_dl_output[0][0].mean()=};
            lava mean firing rate {lava_output[0].mean()=}"""


if __name__ == '__main__':
    unittest.main()
