'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
from lava.lib.dl.slayer.state_space_models.sCIFAR.networks import SCIFARNetworkTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import quantization

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from torch.utils.tensorboard import SummaryWriter
#from s4_original import S4D
from lava.lib.dl.slayer.state_space_models.s4 import S4D
from tqdm.auto import tqdm

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=30, type=float, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')

# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=1, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


run_name = "test_run"

writer = SummaryWriter("runs/" + run_name)
# Data
print(f'==> Preparing {args.dataset} data..')

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

if args.dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(3, 1024).t())
    ])

    # S4 is trained on sequences with no data augmentation!
    transform_train = transform_test = transform

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=False, download=True, transform=transform_test)

    d_input = 3
    d_output = 10



# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


# Model
print('==> Building model..')
model = SCIFARNetworkTorch(
    d_input=3,
    d_output=10,
    d_model=128,
    dropout=0.,
    lr = 0.01,
    d_state=64,
    n_layers=4,
    s4d_exp=12,
    is_real=False,
    get_last=True,
    quantize=False,
)
# model.forward = model.forward_step
#model.train()
#model.encoder.qconfig = torch.quantization.default_qat_qconfig
#model.ff_layers[0].qconfig = torch.quantization.default_qat_qconfig
#model.decoder.qconfig = torch.quantization.default_qat_qconfig
#torch.quantization.prepare_qat(model, inplace=True);
model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

def store_original_params_pre_hook(module, _):
    module.original_params = [p.detach().clone() for p in module.parameters()]


def restore_original_params_hook(module, *_):
    for p, original_p in zip(module.parameters(), module.original_params):
        p.data = original_p.data

def clamp_activations_hook(module, input, output):
    # Define the range for 24 signed bits
    min_val = -2**23
    max_val = 2**23 - 1
    # Clamp the output activations
    output.data = torch.clamp(output.data, min_val, max_val)

# # Register hooks
# for layer in model.modules():
#     if isinstance(layer, S4D):
#         layer.register_forward_pre_hook(store_original_params_pre_hook)
#         layer.register_forward_hook(restore_original_params_hook)


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

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################
inp_scale = 2 ** 9
# Training
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = ((inputs * inp_scale).int() / inp_scale).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    
        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )
    writer.add_scalar("Accuracy/train", correct/total, epoch)
    writer.add_scalar("Loss/train", train_loss/(batch_idx+1), epoch)


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc_ %.3f' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc)
            )
    writer.add_scalar("Accuracy/testing", correct/total, epoch)
    writer.add_scalar("Loss/testing", eval_loss/batch_idx+1, epoch)

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            print("saving checkoint")
            torch.save(state, './checkpoint/eight_states_complex.pth')
            best_acc = acc

        return acc

pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
     if epoch == 0:
         pbar.set_description('Epoch: %d' % (epoch))
     else:
         pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
     train(epoch)
     val_acc = eval(epoch, valloader, checkpoint=True)
     eval(epoch, testloader)
     scheduler.step()
#     # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

for layer in model.modules():
    layer.register_forward_hook(clamp_activations_hook)

checkpoint = torch.load('./checkpoint/eight_states_complex.pth')
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.forward = model.forward_step
for layer in model.s4_layers:
    layer.layer.kernel.quantize = True

train(0)

state = {
         'model': model.state_dict(),
         'acc': 0.,
         'epoch': epoch,
        }

print("saving checkoint")
torch.save(state, './checkpoint/eight_states_quantized_complex.pth')

# inp = next(iter(trainloader))[0].cuda()

# out_c = model(inp)
# out_c.sum().backward()
# grad_C = model.s4_layers[0].layer.kernel.C.grad


# model_s = SCIFARNetworkTorch(
#     d_input=3,
#     d_output=10,
#     d_model=128,
#     dropout=0.,
#     lr = 0.01,
#     d_state=64,
#     n_layers=4,
#     s4d_exp=12,
#     is_real=False,
#     get_last=True,
#     quantize=True,
# ).cuda()

# checkpoint = torch.load('./checkpoint/eight_states_quantized_complex.pth')
# state_dict = checkpoint['model']
# model_s.load_state_dict(state_dict)
# model_s.s4_layers[0].layer.kernel.quantize = True
# model_s.train()

# out_s = model_s.forward_step(inp)

# out_s.sum().backward()
# grad_C_s = model_s.s4_layers[0].layer.kernel.C.grad

# for name, param in model.named_parameters():
#     if param.grad is not None:
#         print(f"Gradient of {name}:")
#         print(param.grad)
#     else:
#         print(f"No gradient for {name}")

# model.forward = model.forward_step
# model.train()
# train(0)
# state = {
#     'model': model.state_dict(),
#     'acc': best_acc, # TODO thats wrong but we dont have the best acc here
#     'epoch': epoch,
# }
# if not os.path.isdir('checkpoint'):
#     os.mkdir('checkpoint')
# print("saving checkoint")
# torch.save(state, './checkpoint/eight_states_quantized_complex.pth')


# todo clamping
# dropout ist null wollen wir das?
# 