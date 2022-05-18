import os
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer
import lava.lib.dl.bootstrap as bootstrap


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 1, # this must be 1 to use batchnorm
                'voltage_decay' : 0.03,
                'tau_grad'      : 1,
                'scale_grad'    : 1,
            }
        neuron_params_drop = {
                **neuron_params, 
                # 'dropout' : slayer.neuron.Dropout(p=0.05),
                # 'norm'    : slayer.neuron.norm.MeanOnlyBatchNorm,
            }
        
        self.blocks = torch.nn.ModuleList([
                # enable affine transform at input
                bootstrap.block.cuba.Input(neuron_params, weight=1, bias=0), 
                bootstrap.block.cuba.Dense(
                    neuron_params_drop, 28*28, 512, 
                    weight_norm=True, weight_scale=2
                ),
                bootstrap.block.cuba.Dense(
                    neuron_params_drop, 512, 512, 
                    weight_norm=True, weight_scale=2
                ),
                bootstrap.block.cuba.Affine(
                    neuron_params, 512, 10, 
                    weight_norm=True, weight_scale=2
                ),
            ])

    def forward(self, x, mode):
        count = []
        N, C, H, W = x.shape
        if mode.base_mode == bootstrap.Mode.ANN:
            x = x.reshape([N, C, H, W, 1])
        else:
            x = slayer.utils.time.replicate(x, 16)

        x = x.reshape(N, -1, x.shape[-1])

        for block, m in zip(self.blocks, mode):
            x = block(x, mode=m)
            count.append(torch.mean(x).item())

        x = torch.mean(x, dim=-1).reshape((N, -1))

        return x, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks \
            if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

trained_folder = 'Trained'
os.makedirs(trained_folder, exist_ok=True)

# device = torch.device('cpu')
device = torch.device('cuda') 

net = Network().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Dataset and dataLoader instances.
training_set = datasets.MNIST(
        root='data/',
        train=True,
        transform=transforms.Compose([
            transforms.RandomAffine(
                degrees=10, 
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]),
        download=True,
    )

testing_set = datasets.MNIST(
        root='data/',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]),
    )

train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(dataset=testing_set , batch_size=32, shuffle=True)

scheduler = bootstrap.routine.Scheduler()

stats = slayer.utils.LearningStats()

epochs = 100
for epoch in range(epochs):
    for i, (input, label) in enumerate(train_loader, 0):
        net.train()
        mode = scheduler.mode(epoch, i, net.training)

        input = input.to(device)

        output, count = net.forward(input, mode)

        loss = F.cross_entropy(output, label.to(device))
        prediction = output.data.max(1, keepdim=True)[1].cpu().flatten()

        stats.training.num_samples += len(label)
        stats.training.loss_sum += loss.cpu().data.item() * input.shape[0]
        stats.training.correct_samples += torch.sum( 
                prediction == label
            ).data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        header = [str(mode)]
        header += [
                'Event rate : ' + ', '.join(
                [f'{c.item():.4f}' for c in count.flatten()]
            )]
        stats.print(epoch, iter=i, header=header, dataloader=train_loader)

    for i, (input, label) in enumerate(test_loader, 0):
        net.eval()
        mode = scheduler.mode(epoch, i, net.training)

        with torch.no_grad():
            input = input.to(device)

            output, count = net.forward(
                    input, 
                    mode=scheduler.mode(epoch, i, net.training)
                )

            loss = F.cross_entropy(output, label.to(device))
            prediction = output.data.max(1, keepdim=True)[1].cpu().flatten()

        stats.testing.num_samples += len(label)
        stats.testing.loss_sum += loss.cpu().data.item() * input.shape[0]
        stats.testing.correct_samples += torch.sum(
                prediction == label
            ).data.item()

        header = [str(mode)]
        header += [
                'Event rate : ' + ', '.join(
                [f'{c.item():.4f}' for c in count.flatten()]
            )]
        stats.print(epoch, iter=i, header=header, dataloader=test_loader)

    if mode.base_mode == bootstrap.routine.Mode.SNN:
        stats.new_line()

    if stats.testing.best_accuracy:
        torch.save(net.state_dict(), trained_folder + '/network.pt')
    stats.update()
    stats.save(trained_folder + '/')
    stats.plot(path=trained_folder + '/')
    net.grad_flow(trained_folder + '/')

