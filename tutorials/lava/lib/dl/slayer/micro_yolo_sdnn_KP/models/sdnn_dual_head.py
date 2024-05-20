import torch
import h5py
import torch.nn.functional as F
from lava.lib.dl import slayer
import numpy as np
import matplotlib.pyplot as plt
import sys
path = ['/home/dbendaya/work/ContinualLearning/tinyYolov3_lava/YOLOsdnn/']
sys.path.extend(path)
from yolo_base import YOLOBase

from .model_utils import quantize_8bit, quantize_5bit, event_rate, SparsityMonitor

#### sdnn_dual_head_yolo model

class Network(YOLOBase):
    def __init__(self,
                 num_classes=80,
                 anchors=[
                     [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
                     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
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

        # quantizer = lambda x: x
        quantizer = quantize_8bit

        self.input_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
        ])

        synapse_kwargs = dict(weight_norm=False, pre_hook_fx=quantizer)
        block_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantizer)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}
        self.backend_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,   3,  16, 3, padding=1, stride=2, weight_scale=1, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  16,  32, 3, padding=1, stride=2, weight_scale=1, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  32,  64, 3, padding=1, stride=2, weight_scale=1, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  64, 128, 3, padding=1, stride=2, weight_scale=3, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 128, 256, 3, padding=1, stride=1, weight_scale=3, **block_kwargs),
        ])

        self.head1_backend = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs,  256,  256, 3, padding=1, stride=2, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  256,  512, 3, padding=1, stride=1, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs,  512, 1024, 3, padding=1, stride=1, **block_kwargs),
            slayer.block.sigma_delta.Conv(neuron_kwargs, 1024,  256, 1, padding=0, stride=1, **block_kwargs),
        ])

        self.head1_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 512, 3, padding=1, stride=1, **block_kwargs),
            slayer.synapse.Conv(512, self.num_output, 1, padding=0, stride=1, **synapse_kwargs),
            slayer.dendrite.Sigma(),
        ])

        self.head2_backend = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs, 256, 128, 1, padding=0, stride=1, **block_kwargs),
            slayer.block.sigma_delta.Unpool(sdnn_params, kernel_size=2, stride=2, **block_kwargs),
        ])

        self.head2_blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Conv(neuron_kwargs, 384, 256, 3, padding=1, stride=1, **block_kwargs),
            slayer.synapse.Conv(256, self.num_output, 1, padding=0, stride=1, **synapse_kwargs),
            slayer.dendrite.Sigma(),
        ])

        # for blk, bias in zip(self.backend_blocks, [0, 0, 0.01, 0.01, 0.01]):
        #     blk.synapse.weight.data += bias
        # for blk, scale in zip(self.backend_blocks, [1, 1, 1, 2, 3]):
        #     blk.synapse.weight.data *= scale
        # for blk, bias in zip(self.head1_backend, [0, 0, 0.01, 0.01]):
        #     blk.synapse.weight.data += bias
        # for blk, scale in zip(self.head1_backend, [1, 0.1, 0.1, 1]):
        #     blk.synapse.weight.data *= scale
        # for blk, bias in zip(self.head2_backend, [0.01]):
        #     blk.synapse.weight.data += bias
        # for blk, scale in zip(self.head2_backend, [0.1]):
        #     blk.synapse.weight.data *= scale
        # self.head2_backend[0].synapse.weight.data += 0.01
        # self.head1_blocks[0].synapse.weight.data *= 3
        # self.head1_blocks[1].weight.data *= 3
        # self.head1_blocks[0].synapse.weight.data += 0.01
        # self.head1_blocks[1].weight.data += 0.01
        # self.head2_blocks[0].synapse.weight.data += 0.005
        # self.head2_blocks[1].weight.data += 0.005

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
        
        backend = input
        for block in self.backend_blocks:
            backend = block(backend)
            count.append(event_rate(backend))
            if has_sparisty_loss:
                sparsity_monitor.append(backend)


        h1_backend = backend        
        for block in self.head1_backend:
            h1_backend = block(h1_backend)
            count.append(event_rate(h1_backend))
            if has_sparisty_loss:
                sparsity_monitor.append(h1_backend)

        head1 = h1_backend        
        for block in self.head1_blocks:
            head1 = block(head1)
            count.append(event_rate(head1))
            if has_sparisty_loss and isinstance(block, slayer.block.sigma_delta.Conv):
                sparsity_monitor.append(head1)

        h2_backend = h1_backend
        for block in self.head2_backend:
            h2_backend = block(h2_backend)
            count.append(event_rate(h2_backend))
            if has_sparisty_loss:
                sparsity_monitor.append(h2_backend)

        head2 = torch.concat([h2_backend, backend], dim=1)
        for block in self.head2_blocks:
            head2 = block(head2)
            count.append(event_rate(head2))
            if has_sparisty_loss and isinstance(block, slayer.block.sigma_delta.Conv):
                sparsity_monitor.append(head2)

        head1 = self.yolo_raw(head1)
        head2 = self.yolo_raw(head2)

        if not self.training:
            output = torch.concat([self.yolo(head1, self.anchors[0]),
                                   self.yolo(head2, self.anchors[1])], dim=1)
        else:
            output = [head1, head2]
        
        return output, torch.FloatTensor(count).reshape((1, -1)).to(input.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        def block_grad_norm(blocks):
            return [b.synapse.grad_norm
                    for b in blocks if hasattr(b, 'synapse')
                    and b.synapse.weight.requires_grad]
        
        grad = block_grad_norm(self.backend_blocks)
        grad += block_grad_norm(self.head1_backend)
        grad += block_grad_norm(self.head2_backend)
        grad += block_grad_norm(self.head1_blocks)
        grad += block_grad_norm(self.head2_blocks)

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

        for i in range(len(self.backend_blocks)):
            self.backend_blocks[i].neuron.bias.data = saved_model[f'backend_blocks.{i}.neuron.bias'].data
            self.backend_blocks[i].neuron.norm.running_mean.data = saved_model[f'backend_blocks.{i}.neuron.norm.running_mean'].data
            self.backend_blocks[i].neuron.delta.threshold.data = saved_model[f'backend_blocks.{i}.neuron.delta.threshold'].data
            self.backend_blocks[i].synapse.weight_g.data = saved_model[f'backend_blocks.{i}.synapse.weight_g'].data
            self.backend_blocks[i].synapse.weight_v.data = saved_model[f'backend_blocks.{i}.synapse.weight_v'].data
            model_keys[f'backend_blocks.{i}.neuron.bias'] = True
            model_keys[f'backend_blocks.{i}.neuron.norm.running_mean'] = True
            model_keys[f'backend_blocks.{i}.neuron.delta.threshold'] = True
            model_keys[f'backend_blocks.{i}.synapse.weight_g'] = True
            model_keys[f'backend_blocks.{i}.synapse.weight_v'] = True

        for i in range(len(self.head1_backend)):
            self.head1_backend[i].neuron.bias.data = saved_model[f'head1_backend.{i}.neuron.bias'].data
            self.head1_backend[i].neuron.norm.running_mean.data = saved_model[f'head1_backend.{i}.neuron.norm.running_mean'].data
            self.head1_backend[i].neuron.delta.threshold.data = saved_model[f'head1_backend.{i}.neuron.delta.threshold'].data
            self.head1_backend[i].synapse.weight_g.data = saved_model[f'head1_backend.{i}.synapse.weight_g'].data
            self.head1_backend[i].synapse.weight_v.data = saved_model[f'head1_backend.{i}.synapse.weight_v'].data
            model_keys[f'head1_backend.{i}.neuron.bias'] = True
            model_keys[f'head1_backend.{i}.neuron.norm.running_mean'] = True
            model_keys[f'head1_backend.{i}.neuron.delta.threshold'] = True
            model_keys[f'head1_backend.{i}.synapse.weight_g'] = True
            model_keys[f'head1_backend.{i}.synapse.weight_v'] = True
        
        for i in range(len(self.head2_backend)):
            self.head2_backend[i].neuron.bias.data = saved_model[f'head2_backend.{i}.neuron.bias'].data
            self.head2_backend[i].neuron.delta.threshold.data = saved_model[f'head2_backend.{i}.neuron.delta.threshold'].data
            self.head2_backend[i].synapse.weight_g.data = saved_model[f'head2_backend.{i}.synapse.weight_g'].data
            self.head2_backend[i].synapse.weight_v.data = saved_model[f'head2_backend.{i}.synapse.weight_v'].data
            model_keys[f'head2_backend.{i}.neuron.bias'] = True
            model_keys[f'head2_backend.{i}.neuron.delta.threshold'] = True
            model_keys[f'head2_backend.{i}.synapse.weight_g'] = True
            model_keys[f'head2_backend.{i}.synapse.weight_v'] = True

            if f'head2_backend.{i}.neuron.norm.running_mean' in saved_model.keys():
                print('Detected different num_outputs.')
                self.head2_backend[i].neuron.norm.running_mean.data = saved_model[f'head2_backend.{i}.neuron.norm.running_mean'].data
                model_keys[f'head2_backend.{i}.neuron.norm.running_mean'] = True
        
        i = 0
        self.head1_blocks[i].neuron.bias.data = saved_model[f'head1_blocks.{i}.neuron.bias'].data
        self.head1_blocks[i].neuron.norm.running_mean.data = saved_model[f'head1_blocks.{i}.neuron.norm.running_mean'].data
        self.head1_blocks[i].neuron.delta.threshold.data = saved_model[f'head1_blocks.{i}.neuron.delta.threshold'].data
        self.head1_blocks[i].synapse.weight_g.data = saved_model[f'head1_blocks.{i}.synapse.weight_g'].data
        self.head1_blocks[i].synapse.weight_v.data = saved_model[f'head1_blocks.{i}.synapse.weight_v'].data
        model_keys[f'head1_blocks.{i}.neuron.bias'] = True
        model_keys[f'head1_blocks.{i}.neuron.norm.running_mean'] = True
        model_keys[f'head1_blocks.{i}.neuron.delta.threshold'] = True
        model_keys[f'head1_blocks.{i}.synapse.weight_g'] = True
        model_keys[f'head1_blocks.{i}.synapse.weight_v'] = True
        
        self.head2_blocks[i].neuron.bias.data = saved_model[f'head2_blocks.{i}.neuron.bias'].data
        self.head2_blocks[i].neuron.norm.running_mean.data = saved_model[f'head2_blocks.{i}.neuron.norm.running_mean'].data
        self.head2_blocks[i].neuron.delta.threshold.data = saved_model[f'head2_blocks.{i}.neuron.delta.threshold'].data
        self.head2_blocks[i].synapse.weight_g.data = saved_model[f'head2_blocks.{i}.synapse.weight_g'].data
        self.head2_blocks[i].synapse.weight_v.data = saved_model[f'head2_blocks.{i}.synapse.weight_v'].data
        model_keys[f'head2_blocks.{i}.neuron.bias'] = True
        model_keys[f'head2_blocks.{i}.neuron.norm.running_mean'] = True
        model_keys[f'head2_blocks.{i}.neuron.delta.threshold'] = True
        model_keys[f'head2_blocks.{i}.synapse.weight_g'] = True
        model_keys[f'head2_blocks.{i}.synapse.weight_v'] = True

        if self.head1_blocks[1].weight.data.shape == saved_model[f'head1_blocks.1.weight'].data.shape:
            self.head1_blocks[1].weight.data = saved_model[f'head1_blocks.1.weight'].data
            self.head2_blocks[1].weight.data = saved_model[f'head2_blocks.1.weight'].data
            model_keys[f'head1_blocks.1.weight'] = True
            model_keys[f'head2_blocks.1.weight'] = True

        residue_keys = [k for k, v in model_keys.items() if v is False]
        if residue_keys:
            for rk in residue_keys:
                print(f'Model parameter {rk} was not loaded.')
