# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

class OxfordDataset(Dataset):
    def __init__(self):
        super(OxfordDataset, self).__init__()
        self.input  = slayer.io.read_1d_spikes('input.bs1' )
        self.target = slayer.io.read_1d_spikes('output.bs1')
        self.target.t = self.target.t.astype(int)

    def __getitem__(self, _):
        return (
            self.input.fill_tensor(torch.zeros(1, 1, 200, 2000)).squeeze(),  # input
            self.target.fill_tensor(torch.zeros(1, 1, 200, 2000)).squeeze(), # target
        )

    def __len__(self):
        return 1 # just one sample for this problem

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
                'threshold'     : 0.1,
                'current_decay' : 1,
                'voltage_decay' : 0.1,
                'requires_grad' : True,     
            }
        
        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(neuron_params, 200, 256),
                slayer.block.cuba.Dense(neuron_params, 256, 200),
            ])
    
    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

if __name__ == '__main__':
    trained_folder = 'Trained'
    os.makedirs(trained_folder, exist_ok=True)

    # device = torch.device('cpu')
    device = torch.device('cuda') 

    net = Network().to(device)

    training_set = OxfordDataset()
    train_loader = DataLoader(dataset=training_set, batch_size=1)

    for i, (input, target) in enumerate(train_loader):
        output = net(input.to(device))

    # net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    net = slayer.auto.SequentialNetwork(trained_folder + '/network.net').to(device)
    output = net(input.to(device))
    event = slayer.io.tensor_to_event(output.cpu().data.numpy())
    # event.show(plt.figure(figsize=(10, 10)))
    plt.plot(event.t, event.x, '.')
    plt.show()