# Copyright : (c) UC Regents, Emre Neftci, 2022 Intel Corporation
# Licence : GPLv3

import os
import torch
from torch.utils.data import DataLoader

import lava.lib.dl.decolle as decolle
import lava.lib.dl.slayer as slayer

from tutorials.lava.lib.dl.slayer.nmnist.nmnist import NMNISTDataset, augment


class DECOLLENetwork(torch.nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape, burn_in=0):

        super(DECOLLENetwork, self).__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3.,
            'requires_grad': True,
            'persistent_state': True
        }
        neuron_params_drop = {**neuron_params}

        self.burn_in = burn_in
        self.blocks = torch.nn.ModuleList()
        self.readout_layers = torch.nn.ModuleList()

        hidden_shape = [input_shape] + hidden_shape
        for i in range(len(hidden_shape)-1):
            self.blocks.append(slayer.block.cuba.Dense(
                neuron_params_drop, hidden_shape[i], hidden_shape[i+1],
                weight_norm=False)
            )

            # One fixed readout per layer
            readout = torch.nn.Linear(hidden_shape[i+1],
                                      output_shape,
                                      bias=False)
            readout.weight.requires_grad = False
            self.readout_layers.append(readout)


    def forward(self, spike):
        spike.requires_grad_()
        spikes = []
        readouts = []
        voltages = []
        count = []

        for block in self.blocks:
            # Decompose the behavior of the block to obtain the voltage
            # for the regularization
            z = block.synapse(spike.detach())
            _, voltage = block.neuron.dynamics(z)
            voltages.append(voltage)

            spike = block.neuron.spike(voltage)
            spikes.append(spike)
            count.append(torch.mean(spike.detach()))

        for ro, spike in zip(self.readout_layers, spikes):
            # Compute readouts
            readout = []
            for t in range(spike.shape[-1]):
                readout.append(ro(spike[..., t]))
            readouts.append(torch.stack(readout, dim=-1))

        return spikes, readouts, voltages, count

    def init_state(self, inputs, burn_in=None):
        self.reset_()
        # initialize state + crop inputs
        if burn_in is None:
            burn_in = self.burn_in

        self.forward(inputs[..., :burn_in])
        return inputs[..., burn_in:]

    def reset_(self):
        # reset the state after each example
        for block in self.blocks:
            block.neuron.current_state[:] = 0.
            block.neuron.voltage_state[:] = 0.


if __name__ == '__main__':
    trained_folder = 'Trained'
    os.makedirs(trained_folder, exist_ok=True)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    net = DECOLLENetwork(input_shape=34 * 34 * 2,
                         hidden_shape=[512, 256],
                         output_shape=10,
                         burn_in=10).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    training_set = NMNISTDataset(train=True,
                                 transform=augment, download=True)
    testing_set = NMNISTDataset(train=False,
                                )

    train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=32, shuffle=True)

    error = decolle.loss.DECOLLELoss(torch.nn.CrossEntropyLoss,
                                    reg=0.01, reduction='mean')

    stats = slayer.utils.LearningStats()
    assistant = decolle.utils.DECOLLEAssistant(
        net, error, optimizer, stats,
        classifier=slayer.classifier.Rate.predict, count_log=True
    )

    epochs = 200

    for epoch in range(epochs):
        for i, (input, label) in enumerate(train_loader):  # training loop
            output, count = assistant.train(input, label)
            header = [
                'Event rate : ' +
                ', '.join([f'{c.item():.4f}' for c in count.flatten()])
            ]
            stats.print(epoch, iter=i, header=header, dataloader=train_loader)

        for i, (input, label) in enumerate(test_loader):  # training loop
            output, count = assistant.test(input, label)
            header = [
                'Event rate : ' +
                ', '.join([f'{c.item():.4f}' for c in count.flatten()])
            ]
            stats.print(epoch, iter=i, header=header, dataloader=test_loader)

        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), trained_folder + '/network.pt')
        stats.update()
        stats.save(trained_folder + '/')
        stats.plot(path=trained_folder + '/')
