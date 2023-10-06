# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lava.lib.dl import slayer

from ..yolo_base import YOLOBase

"""Tiny YOLOv3 network for 8 chip Loihi form factor (Kapoho Point). It is a
modified version of Tiny YOLOv3 architecture with redesign of the layers to
best mapping to the Loihi 2 KP form factor."""


class Network(YOLOBase):
    """Sigma-Delta YOLO-KP network definition.

    Parameters
    ----------
    num_classes : int, optional
        Number of object classes to predict, by default 80.
    anchors : List[List[Tuple[float, float]]], optional
        Prediction anchor points.
    threshold : float, optional
        Sigma-delta neuron threshold, by default 0.1.
    tau_grad : float, optional
        Surrogate gradient relaxation parameter, by default 0.1.
    scale_grad : float, optional
        Surrogate gradient scale parameter, by default 0.1.
    clamp_max : float, optional
        Maximum clamp value for converting raw prediction to bounding box
        prediciton. This is useful for improving stability of the training.
        By default 5.0.
    """
    def __init__(self,
                 num_classes: int = 80,
                 anchors: List[List[Tuple[float, float]]] = [
                     [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
                 ],
                 threshold: float = 0.1,
                 tau_grad: float = 0.1,
                 scale_grad: float = 0.1,
                 clamp_max: float = 5.0) -> None:
        super().__init__(num_classes=num_classes,
                         anchors=anchors,
                         clamp_max=clamp_max)

        sigma_params = {  # sigma-delta neuron parameters
            'threshold'     : threshold,   # delta unit threshold
            'tau_grad'      : tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,       # trainable threshold
            'shared_param'  : True,        # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : F.relu,      # activation function
        }

        # standard imagenet normalization of RGB images
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1, 1])
        self.normalize_std  = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1, 1])

        self.input_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
        ])

        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        def quantize_5bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        synapse_kwargs = dict(weight_norm=False, pre_hook_fx=quantize_8bit)
        block_5_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_5bit)
        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        self.input_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
        ])

        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,   3,  16, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  16,  32, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  32,  64, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  64, 128, 3, padding=1, stride=2, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 128, 256, 3, padding=1, stride=1, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 256, 3, padding=1, stride=2, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 512, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 512, 256, 1, padding=0, stride=1, weight_scale=3, **block_5_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 512, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
        ])

        self.heads = torch.nn.ModuleList([
            slayer.synapse.Conv(512, self.num_output, 1, padding=0, stride=1, **synapse_kwargs),
            slayer.dendrite.Sigma(),
        ])

    def forward(
        self,
        input: torch.tensor,
        sparsity_monitor: slayer.loss.SparsityEnforcer = None
    ) -> Tuple[Union[torch.tensor, List[torch.tensor]], torch.tensor]:
        """Forward computation step of the network module.

        Parameters
        ----------
        input : torch.tensor
            Input frames tensor.
        sparsity_monitor : slayer.loss.SparsityEnforcer, optional
            Sparsity monitor module. If None, sparisty is not enforced.
            By default None.

        Returns
        -------
        Union[torch.tensor, List[torch.tensor]]
            Output of the network.

            * If the network is in training mode, the output is a list of
            raw output tensors of the different heads of the network.
            * If the network is in testing mode, the output is the consolidated
            prediction bounding boxes tensor.

            Note: the difference in the output behavior is done to apply
            loss to the raw tensor for better training stability.
        torch.tensor
            Event rate statistics.
        """
        has_sparisty_loss = sparsity_monitor is not None
        if self.normalize_mean.device != input.device:
            self.normalize_mean = self.normalize_mean.to(input.device)
            self.normalize_std = self.normalize_std.to(input.device)
        input = (input - self.normalize_mean) / self.normalize_std

        count = []
        for block in self.input_blocks:
            input = block(input)
            count.append(slayer.utils.event_rate(input))

        x = input
        for block in self.blocks:
            x = block(x)
            count.append(slayer.utils.event_rate(x))
            if has_sparisty_loss:
                sparsity_monitor.append(x)

        for head in self.heads:
            x = head(x)
            count.append(torch.mean(torch.abs((x) > 0).to(x.dtype)).item())

        head1 = self.yolo_raw(x)

        if not self.training:
            output = self.yolo(head1, self.anchors[0])
        else:
            output = [head1]

        return (output,
                torch.FloatTensor(count).reshape((1, -1)).to(input.device))

    def export_hdf5_head(self, handle: h5py.Dataset) -> None:
        """Exports the hdf5 description of the head of the network. This is
        done because the head does not follow the slayer.block construct.

        Parameters
        ----------
        handle : h5py.Dataset
            Hdf5 file handle.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape).cpu().data.numpy()

        handle.create_dataset(
            'type', (1, ), 'S10', ['conv'.encode('ascii', 'ignore')]
        )
        synapse = self.heads[0]
        neuron = self.heads[1]
        handle.create_dataset('shape', data=np.array(neuron.shape))
        handle.create_dataset('inChannels', data=synapse.in_channels)
        handle.create_dataset('outChannels', data=synapse.out_channels)
        handle.create_dataset('kernelSize', data=synapse.kernel_size[:-1])
        handle.create_dataset('stride', data=synapse.stride[:-1])
        handle.create_dataset('padding', data=synapse.padding[:-1])
        handle.create_dataset('dilation', data=synapse.dilation[:-1])
        handle.create_dataset('groups', data=synapse.groups)
        handle.create_dataset('weight', data=weight(synapse))

        device_params = neuron.device_params
        device_params['sigma_output'] = True
        for key, value in device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)

    def export_hdf5(self, filename: str) -> None:
        """Export the YOLO-KP network as hdf5 description.

        Parameters
        ----------
        filename : str
            Filename of the model description.
        """
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        offset = 0
        for i, b in enumerate(self.input_blocks):
            b.export_hdf5(layer.create_group(f'{i + offset}'))
        offset += len(self.input_blocks)
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i + offset}'))
        offset += len(self.blocks)
        self.export_hdf5_head(layer.create_group(f'{offset}'))

    def grad_flow(self, path: str) -> List[torch.tensor]:
        """Montiors gradient flow along the layers.

        Parameters
        ----------
        path : str
            Path for output plot export.

        Returns
        -------
        List[torch.tensor]
            List of gradient norm per layer.
        """
        # helps monitor the gradient flow
        def block_grad_norm(blocks):
            return [b.synapse.grad_norm
                    for b in blocks if hasattr(b, 'synapse')
                    and b.synapse.weight.requires_grad]

        grad = block_grad_norm(self.blocks)

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def load_model(self, model_file: str) -> None:
        """Selectively loads the model from save pytorch state dictionary.
        If the number of output layer does not match, it will ignore the last
        layer in the head. Other states should match, if not, there might be
        some mismatch with the model file.

        Parameters
        ----------
        model_file : str
            Path to pytorch model file.
        """
        saved_model = torch.load(model_file)
        model_keys = {k : False for k in saved_model.keys()}
        device = self.anchors.device
        self.anchors.data = saved_model['anchors'].data.to(device)
        self.input_blocks[0].neuron.bias.data = saved_model[f'input_blocks.0.neuron.bias'].data.to(device)
        self.input_blocks[0].neuron.delta.threshold.data = saved_model[f'input_blocks.0.neuron.delta.threshold'].data.to(device)
        model_keys[f'anchors'] = True
        model_keys[f'input_blocks.0.neuron.bias'] = True
        model_keys[f'input_blocks.0.neuron.delta.threshold'] = True

        for i in range(len(self.blocks)):
            self.blocks[i].neuron.bias.data = saved_model[f'blocks.{i}.neuron.bias'].data
            self.blocks[i].neuron.norm.running_mean.data = saved_model[f'blocks.{i}.neuron.norm.running_mean'].data
            self.blocks[i].neuron.delta.threshold.data = saved_model[f'blocks.{i}.neuron.delta.threshold'].data
            self.blocks[i].synapse.weight_g.data = saved_model[f'blocks.{i}.synapse.weight_g'].data
            self.blocks[i].synapse.weight_v.data = saved_model[f'blocks.{i}.synapse.weight_v'].data
            model_keys[f'blocks.{i}.neuron.bias'] = True
            model_keys[f'blocks.{i}.neuron.norm.running_mean'] = True
            model_keys[f'blocks.{i}.neuron.delta.threshold'] = True
            model_keys[f'blocks.{i}.synapse.weight_g'] = True
            model_keys[f'blocks.{i}.synapse.weight_v'] = True

        if self.heads[0].weight.data.shape == saved_model[f'heads.0.weight'].data.shape:
            self.heads[0].weight.data = saved_model[f'heads.0.weight'].data
            model_keys[f'heads.0.weight'] = True

        residue_keys = [k for k, v in model_keys.items() if v is False]
        if residue_keys:
            for rk in residue_keys:
                print(f'Model parameter {rk} was not loaded.')
