# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import os
import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_conditions import RunSteps

from lava.lib.dl import netx
from lava.proc import io

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
    import logging
    from lava.magma.compiler.subcompilers.nc.ncproc_compiler import \
        CompilerOptions
    from lava.magma.core.run_configs import Loihi2HwCfg
    from lava.proc import embedded_io as eio
    skip_loihi_test = False
    skip_message_loihi = ''
else:
    skip_message_loihi = 'Loihi2 compiler is not available in this system.'
    skip_loihi_test = True

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False
HAVE_DISPLAY = 'DISPLAY' in os.environ
root = os.path.dirname(os.path.abspath(__file__))

if not skip_loihi_test:
    CompilerOptions.verbose = verbose


class TestHdf5NetxNx(unittest.TestCase):
    @unittest.skipIf(skip_loihi_test, skip_message_loihi)
    def test_tinynet(self) -> None:
        """Tests the output of three layer CNN."""
        steps_per_sample = 16
        net = netx.hdf5.Network(net_config=root + '/tiny.net',
                                num_layers=3,
                                reset_interval=steps_per_sample)
        num_steps = steps_per_sample + len(net)
        out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape)
        out_logger = io.sink.RingBuffer(shape=net.out.shape, buffer=num_steps)

        # set input bias for the ground truth sample
        net.in_layer.neuron.bias_mant.init = np.load(
            root + '/gts/tinynet/input_bias.npy').astype(int)

        net.out.connect(out_adapter.inp)
        out_adapter.out.connect(out_logger.a_in)

        pre_run_fxs = []
        post_run_fxs = []

        if verbose:
            print(net)
            net._log_config.level = logging.INFO

        net.run(condition=RunSteps(num_steps=num_steps),
                run_cfg=Loihi2HwCfg(pre_run_fxs=pre_run_fxs,
                                    post_run_fxs=post_run_fxs))
        output = out_logger.data.get()
        net.stop()

        gt = np.load(root + '/gts/tinynet/output.npy')[0]
        gt = gt.transpose([2, 1, 0, 3])[..., 1:]

        error = np.abs(output[..., 2:17] - gt).sum()
        if verbose:
            print('Output spike error:', error)
            if HAVE_DISPLAY:
                plt.figure()
                out_ae = np.argwhere(output.squeeze() > 0)
                gt_ae = np.argwhere(gt.squeeze() > 0)
                plt.plot(gt_ae[:, 1] + len(net) - 1,
                         gt_ae[:, 0],
                         '.', markersize=15, label='Ground Truth')
                plt.plot(out_ae[:, 1], out_ae[:, 0], '.', label='Output')
                plt.xlabel('Time')
                plt.ylabel('Neuron ID')
                plt.legend()
                plt.show()

        self.assertTrue(
            error == 0,
            f'Output spike and ground truth do not match. '
            f'Found {output[..., 2:17] = } and {gt = }. '
            f'Error was {error}.')

    @unittest.skipIf(skip_loihi_test, skip_message_loihi)
    def test_pilotnet_lif(self) -> None:
        """Tests the output of PilotNet LIF."""
        # Verifying all layers take a lot of time. So this test verifies
        # only one layer. If that layer failes, it is recommended to
        # decrease idx to find out the layer that has diverged.
        idx = 8  # Layer to verify.
        steps_per_sample = 16
        net_config = root + '/gts/pilotnet_lif/network.net'
        net = netx.hdf5.Network(net_config=net_config,
                                reset_interval=steps_per_sample)
        # num_steps = steps_per_sample
        num_steps = 15
        net.in_layer.neuron.bias_mant.init = np.load(
            root + '/gts/pilotnet_lif/image.npy').astype(int)

        pre_run_fxs = []
        post_run_fxs = []

        if verbose:
            print(net)
            net._log_config.level = logging.INFO
        net.run(condition=RunSteps(num_steps=num_steps),
                run_cfg=Loihi2HwCfg(pre_run_fxs=pre_run_fxs,
                                    post_run_fxs=post_run_fxs))
        if idx == 0:
            result = net.layers[0].neuron.v.get()
        else:
            result = net.layers[idx].neuron.u.get()
        net.stop()

        if idx == 0:
            ground_truth = np.load(root + '/gts/pilotnet_lif/input-v.npy')
        elif idx < 6:
            ground_truth = np.load(root
                                   + f'/gts/pilotnet_lif/conv-{idx - 1}-u.npy')
        else:
            ground_truth = np.load(root
                                   + f'/gts/pilotnet_lif/fc-{idx - 6}-u.npy')

        if verbose:
            result = result.astype(int)
            print(f'{result.shape=}')
            print(f'{ground_truth.shape=}')
            print(f'{result=}'.replace('=', '=\n'))
            print(f'{ground_truth[..., num_steps - 1]=}'.replace('=', '=\n'))
            print(f'{ground_truth[0]=}')
        self.assertTrue(np.array_equal(result,
                                       ground_truth[..., num_steps - 1]))

    @unittest.skipIf(skip_loihi_test, skip_message_loihi)
    def test_pilotnet_lif_spike_input(self) -> None:
        """Tests the output of PilotNet LIF driven by CProc spike injection."""
        # Verifying all layers take a lot of time. So this test verifies
        # only one layer. If that layer failes, it is recommended to
        # decrease idx to find out the layer that has diverged.
        idx = 5  # Layer to verify.
        steps_per_sample = 16
        net_config = root + '/gts/pilotnet_lif/network.net'
        net = netx.hdf5.Network(net_config=net_config,
                                num_layers=5,
                                skip_layers=1,
                                reset_interval=steps_per_sample,
                                reset_offset=1)
        input_spikes = np.load(
            root + '/gts/pilotnet_lif/input-s.npy').astype(int)
        input_spikes = input_spikes
        source = io.source.RingBuffer(data=input_spikes)
        spike_gen = eio.spike.PyToN3ConvAdapter(shape=net.in_layer.inp.shape)

        num_steps = steps_per_sample
        source.s_out.connect(spike_gen.inp)
        spike_gen.out.connect(net.inp)

        pre_run_fxs = []
        post_run_fxs = []

        if verbose:
            print(net)
            net._log_config.level = logging.INFO
        net.run(condition=RunSteps(num_steps=num_steps),
                run_cfg=Loihi2HwCfg(pre_run_fxs=pre_run_fxs,
                                    post_run_fxs=post_run_fxs))
        result = net.out_layer.neuron.u.get()
        net.stop()

        if 0 < idx < 6:
            ground_truth = np.load(root
                                   + f'/gts/pilotnet_lif/conv-{idx - 2}-u.npy')
            self.assertTrue(np.array_equal(result,
                                           ground_truth[..., num_steps - 1]))

    @unittest.skipIf(skip_loihi_test, skip_message_loihi)
    def test_pilotnet_sdnn_square(self) -> None:
        """Tests the output of pilotnet sdnn with reduced x dimension."""
        net_config = root + '/gts/pilotnet_sdnn/square_network.net'
        net = netx.hdf5.Network(net_config=net_config,
                                skip_layers=1)
        if verbose:
            print(net)

        input = np.load(root + '/gts/pilotnet_sdnn/act0.npy')[:33, :33]

        num_steps = 5
        # Loihi execution
        source = io.source.RingBuffer(data=input)
        inp_adapter = eio.spike.PyToN3ConvAdapter(shape=net.inp.shape)
        source.s_out.connect(inp_adapter.inp)
        inp_adapter.out.connect(net.in_layer.synapse.s_in)

        pre_run_fxs = []
        post_run_fxs = []

        # Loihi Execution
        run_cfg = Loihi2HwCfg(pre_run_fxs=pre_run_fxs,
                              post_run_fxs=post_run_fxs)
        run_cnd = RunSteps(num_steps=num_steps)
        if verbose:
            net._log_config.level = logging.INFO
        net.run(condition=run_cnd, run_cfg=run_cfg)
        sigma1 = net.layers[0].neuron.sigma.get()
        sigma2 = net.layers[1].neuron.sigma.get()
        sigma3 = net.layers[2].neuron.sigma.get()
        sigma4 = net.layers[3].neuron.sigma.get()
        net.stop()

        sigma1_gt = np.load(root + '/gts/pilotnet_sdnn/sigma1_square.npy')
        sigma2_gt = np.load(root + '/gts/pilotnet_sdnn/sigma2_square.npy')
        sigma3_gt = np.load(root + '/gts/pilotnet_sdnn/sigma3_square.npy')
        sigma4_gt = np.load(root + '/gts/pilotnet_sdnn/sigma4_square.npy')

        self.assertTrue(np.array_equal(sigma1, sigma1_gt[..., num_steps - 1]))
        self.assertTrue(np.array_equal(sigma2, sigma2_gt[..., num_steps - 1]))
        self.assertTrue(np.array_equal(sigma3, sigma3_gt[..., num_steps - 1]))
        self.assertTrue(np.array_equal(sigma4, sigma4_gt[..., num_steps - 1]))

    @unittest.skipIf(skip_loihi_test, skip_message_loihi)
    def test_pilotnet_sdnn(self) -> None:
        """Tests the output of pilotnet sdnn."""
        net_config = root + '/gts/pilotnet_sdnn/network.net'
        net = netx.hdf5.Network(net_config=net_config,
                                skip_layers=1)
        if verbose:
            print(net)

        input = np.load(root + '/gts/pilotnet_sdnn/act0.npy')

        num_steps = len(net) + 1
        # Loihi execution
        source = io.source.RingBuffer(data=input)
        inp_adapter = eio.spike.PyToN3ConvAdapter(
            shape=net.inp.shape, num_message_bits=16)
        sink = io.sink.RingBuffer(shape=net.out.shape,
                                  buffer=num_steps)
        out_adapter = eio.spike.NxToPyAdapter(
            shape=net.out.shape, num_message_bits=24)

        source.s_out.connect(inp_adapter.inp)
        inp_adapter.out.connect(net.inp)
        net.out.connect(out_adapter.inp)
        out_adapter.out.connect(sink.a_in)

        pre_run_fxs = []
        post_run_fxs = []

        # Loihi Execution
        run_cfg = Loihi2HwCfg(pre_run_fxs=pre_run_fxs,
                              post_run_fxs=post_run_fxs)
        run_cnd = RunSteps(num_steps=num_steps)
        if verbose:
            net._log_config.level = logging.INFO
        net.run(condition=run_cnd, run_cfg=run_cfg)
        # sigma2 = net.layers[1].neuron.sigma.get()
        sigma = net.out_layer.neuron.sigma.get()
        output = sink.data.get().astype(np.int32)
        net.stop()
        output = (output << 8) >> 8

        # Keeping this commented code to debug if something goes wrong in future
        # sigma2_gt = np.load(root + '/gts/pilotnet_sdnn/sigma2.npy')
        # self.assertTrue(np.array_equal(sigma2, sigma2_gt[..., num_steps - 1]))

        gt = np.load(root + '/gts/pilotnet_sdnn/output.npy')
        error = np.abs(output - gt).sum()
        if verbose:
            print('Network:')
            print(net)
            print(f'{output=}')
            print(f'{gt=}')
            print(f'{sigma=}')
            print('PilotNet SDNN spike error:', error)

        self.assertTrue(
            error == 0,
            f'Output spike and ground truth do not match for PilotNet SDNN. '
            f'Found {output[output != gt] = } and {gt[output != gt] = }. '
            f'Error was {error}.'
        )


if __name__ == '__main__':
    unittest.main()
