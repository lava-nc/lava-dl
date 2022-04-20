# Copyright © 2022 Intel Corporation.
# 
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express 
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy, 
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are 
# expressly stated in the License.

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn.functional as F

class Assistant:
    def __init__(self, net, optimizer, stats, lam, device):
        self.net = net
        self.optimizer = optimizer
        self.device = device
        self.stats = stats
        self.lam = lam

    def step_lr(self):
        for param_group in self.optimizer.param_groups:    
            print('\nLearning rate reduction from', param_group['lr'])
            param_group['lr'] /= 10/3
        
    def train(self, input, ground_truth):
        self.net.train()
        
        input = input.to(self.device)
        ground_truth = ground_truth.to(self.device)

        output, event_loss, count = self.net(input)
        loss = F.mse_loss(output.flatten(), ground_truth.flatten()) + self.lam * event_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.stats.training.num_samples += input.shape[0]
        self.stats.training.loss_sum += (loss - self.lam * event_loss).cpu().data.item() * output.shape[0] # save only the mean square value

        return count
    
    def test(self, input, ground_truth):
        self.net.eval()
        
        with torch.no_grad():
            input = input.to(self.device)
            ground_truth = ground_truth.to(self.device)

            output, event_loss, count = self.net(input)
            loss = F.mse_loss(output.flatten(), ground_truth.flatten()) + self.lam * event_loss

            self.stats.testing.num_samples += input.shape[0]
            self.stats.testing.loss_sum += (loss - self.lam * event_loss).cpu().data.item() * output.shape[0] # save only the mean square value

        return count

def compare_ops(net, counts, mse):
    shapes = [b.shape for b in net.blocks if hasattr(b, 'neuron')]

    # synops calculation
    sdnn_synops = []
    ann_synops = []
    for l in range(1, len(net.blocks)):
        if hasattr(net.blocks[l], 'neuron') is False:
            break
        conv_synops = ( # ignoring padding
                counts[l-1]
                * net.blocks[l].synapse.out_channels
                * np.prod(net.blocks[l].synapse.kernel_size)
                / np.prod(net.blocks[l].synapse.stride)
            )
        sdnn_synops.append(conv_synops)
        ann_synops.append(conv_synops*np.prod(net.blocks[l-1].shape)/counts[l-1])
        # ann_synops.append(conv_synops*np.prod(net.blocks[l-1].shape)/counts[l-1]*np.prod(net.blocks[l].synapse.stride))
        
    for l in range(l+1, len(net.blocks)):
        fc_synops = counts[l-2] * net.blocks[l].synapse.out_channels
        sdnn_synops.append(fc_synops)
        ann_synops.append(fc_synops*np.prod(net.blocks[l-1].shape)/counts[l-2])

    # event and synops comparison
    total_events = np.sum(counts)
    total_synops = np.sum(sdnn_synops)
    total_ann_activs = np.sum([np.prod(s) for s in shapes])
    total_ann_synops = np.sum(ann_synops)
    total_neurons = np.sum([np.prod(s) for s in shapes])
    steps_per_inference = 1

    print(f'|{"-"*77}|')
    print('|', ' '*23,                 '|          SDNN           |           ANN           |')
    print(f'|{"-"*77}|')
    print('|', ' '*7, f'|     Shape     |  Events  |    Synops    | Activations|    MACs    |')
    print(f'|{"-"*77}|')
    for l in range(len(counts)):
        print(f'| layer-{l} | ', end='')
        if len(shapes[l]) == 3: z, y, x = shapes[l]
        elif len(shapes[l]) == 1:
            z = shapes[l][0]
            y = x = 1
        print(f'({x:-3d},{y:-3d},{z:-3d}) | {counts[l]:8.2f} | ', end='')
        if l==0:
            print(f'{" "*12} | {np.prod(shapes[l]):-10.0f} | {" "*10} |')
        else:
            print(f'{sdnn_synops[l-1]:12.2f} | {np.prod(shapes[l]):10.0f} | {ann_synops[l-1]:10.0f} |')
    print(f'|{"-"*77}|')
    print(f'|  Total  | {" "*13} | {total_events:8.2f} | {total_synops:12.2f} | {total_ann_activs:10.0f} | {total_ann_synops:10.0f} |')
    print(f'|{"-"*77}|')

    print('\n')
    print(f'MSE            : {mse:.5} sq. radians')
    print(f'Total neurons  : {total_neurons}')
    print(f'Events sparsity: {total_ann_activs/total_events:5.2f}x')
    print(f'Synops sparsity: {total_ann_synops/total_synops:5.2f}x')
