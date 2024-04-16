import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm.auto import tqdm
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import lava.lib.dl.slayer as slayer
import utils
import torch
from network import SCIFARNetwork
#from torch.utils.tensorboard import SummaryWriter

lam = None

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


#def event_rate_loss(x, max_rate=0.01):
#    mean_event_rate = torch.mean(torch.abs(x))
#    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay')
parser.add_argument('--lam', default=0.0000001, type=float, help='Lagrangian for event rate loss')
# Scheduler
parser.add_argument('--epochs', default=100, type=float, help='Training epochs')
parser.add_argument('--old_optimizer', action='store_true')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--skip', action='store_true')
parser.add_argument('--loihi', action='store_true')
parser.add_argument('--early_mean', action='store_true')
parser.add_argument('--disable_quantize', action='store_true')
parser.add_argument('--add_dropout', action='store_true')
parser.add_argument('--num_layers', default=1, type=int, help='Number of layers')


args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
trained_folder = 'Trained'
logs_folder = 'Logs'

os.makedirs(trained_folder, exist_ok=True)
os.makedirs(logs_folder   , exist_ok=True)
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(3, 1024).t())
    ])
transform_train = transform_test = transform

# Datasets
trainset = torchvision.datasets.CIFAR10(
    root='./data/cifar/', train=True, download=True, transform=transform_train)
trainset, _ = split_train_val(trainset, val_split=0.1)

testset = torchvision.datasets.CIFAR10(
    root='./data/cifar/', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader  = torch.utils.data.DataLoader(dataset=testset , batch_size=args.batch_size, shuffle=True, num_workers=8)




device = torch.device('cuda')
#net = Network().to(device)
net = SCIFARNetwork(dropout=args.dropout, num_layers=args.num_layers).to(device)


if args.old_optimizer:
    optimizer = torch.optim.RAdam(net.parameters(), lr=args.lr, weight_decay=1e-5)
else:
    optimizer, scheduler = setup_optimizer(
    net, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)


stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(
        net=net,
        error=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        stats=stats,
        count_log=False,
        lam=None,
        classifier = lambda x : x.argmax(1))

for epoch in range(args.epochs):        
    net.train()
    for i, (input, ground_truth) in enumerate(train_loader): # training loop
        input, ground_truth = input.to(device), ground_truth.to(device)
        assistant.train(input, ground_truth)
        print(f'\r[Epoch {epoch:3d}/{args.epochs}] {stats}', end='')
        #writer.add_scalar("Accuracy/train", stats.training.accuracy, epoch)
        #writer.add_scalar("Loss/train", stats.training.loss, epoch)
    
    net.eval()
    for i, (input, ground_truth) in enumerate(test_loader): # testing loop
        input, ground_truth = input.to(device), ground_truth.to(device)
        assistant.test(input, ground_truth)
        print(f'\r[Epoch {epoch:3d}/{args.epochs}] {stats}', end='')
       
     
    if stats.testing.best_loss:  
        torch.save(net.state_dict(), trained_folder + '/network.pt')
    stats.update()
    stats.save(trained_folder + '/')
    
    # gradient flow monitoring
    net.grad_flow(trained_folder + '/')
    
    # checkpoint saves
    if epoch%10 == 0:
        torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()}, logs_folder + f'/checkpoint{epoch}.pt')    
    