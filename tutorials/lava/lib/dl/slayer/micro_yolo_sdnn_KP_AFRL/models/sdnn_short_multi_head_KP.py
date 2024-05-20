import torch
import h5py
import torch.nn.functional as F
from lava.lib.dl import slayer
import numpy as np
import matplotlib.pyplot as plt
import sys
from yolo_base import YOLOBase

from .model_utils import quantize_8bit, quantize_5bit, event_rate, SparsityMonitor

#### sdnn_single_head_KP_yolo model

class Network(YOLOBase):
    def __init__(self,
                 num_classes=80,
                 anchors=[
                     [[0.4667, 0.7455], [0.4282, 0.1168], [0.1563, 0.7116]],
                     [[0.8314, 0.4178], [0.3581, 0.5125], [0.6074, 0.5301]],
                     [[0.1104, 0.4343], [0.3004, 0.3085], [0.7987, 0.7079]],
                     [[0.7589, 0.152 ], [0.1295, 0.1428], [0.5552, 0.3241]],
                 ],
                 threshold=0.1,
                 tau_grad=0.1,
                 scale_grad=0.1,
                 clamp_max=5.0):
        super().__init__(num_classes=num_classes, anchors=anchors, clamp_max=clamp_max)

        sigma_params = { # sigma-delta neuron parameters
            'threshold'     : threshold,   # delta unit threshold
            'tau_grad'      : tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,  # trainable threshold
            'shared_param'  : True,   # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : F.relu, # activation function
        }

        # standard imagenet normalization of RGB images
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1, 1])
        self.normalize_std  = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1, 1])

        self.input_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
        ])

        synapse_kwargs = dict(weight_norm=False, pre_hook_fx=quantize_8bit)
        block_5_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_5bit)
        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        self.input_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
        ])

        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,   3,  8, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  8,  16, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  16,  32, 3, padding=1, stride=2, weight_scale=1, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  32, 64, 3, padding=1, stride=2, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 64, 128, 3, padding=1, stride=1, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 128, 128, 3, padding=1, stride=2, weight_scale=3, **block_8_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 128, 256, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 128, 1, padding=0, stride=1, weight_scale=3, **block_5_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 128, 256, 3, padding=1, stride=1, weight_scale=3, **block_5_kwargs),
        ])

        self.heads = torch.nn.ModuleList([
            slayer.synapse.Conv(256, self.num_output, 1, padding=0, stride=1, **synapse_kwargs),
            slayer.dendrite.Sigma(),
        ])

    def forward(self, input, sparsity_monitor: SparsityMonitor=None):
        has_sparisty_loss = sparsity_monitor is not None
        if self.normalize_mean.device != input.device:
            self.normalize_mean = self.normalize_mean.to(input.device)
            self.normalize_std = self.normalize_std.to(input.device)
        input = (input - self.normalize_mean) / self.normalize_std

        count = []
        for block in self.input_blocks:
            input = block(input)
            count.append(event_rate(input))

        x = input
        for block in self.blocks:
            x = block(x)
            count.append(event_rate(x))
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

        return output, torch.FloatTensor(count).reshape((1, -1)).to(input.device)

    def export_hdf5_head(self, handle):
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

    def export_hdf5(self, filename):
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
            
    def grad_flow(self, path):
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

    def load_model(self, model_file: str):
        saved_model = torch.load(model_file)
        model_keys = {k:False for k in saved_model.keys()}
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
