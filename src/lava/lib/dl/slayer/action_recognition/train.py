import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model to train on. See model registry in model.py for a list of available models (efficientnet-b0-LSTM, efficientnet-b0-S4D, ..).')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size. Default: 8')
parser.add_argument('--epochs', type=int, default=100, help='Num epochs. Default: 100')
parser.add_argument('--print-interval', type=int, default=100, help='Print information each N batches. Default: 100')
parser.add_argument('--lr', type=float, metavar="LEARNING_RATE", default=1e-3, help='Learning rate. Default: 1e-3')
parser.add_argument('--no-train-backbone', action='store_true', help='Train backbone.')
parser.add_argument('--continue-training', action='store_true', help='Load last checkpoint and continue training.')


# Dataset
parser.add_argument('--dataset', type=str, default="NTU", help='NTU or HARDVS. Default: NTU')
parser.add_argument('--frames-per-sample', type=int, default=30, help='Num frames per sample. Default: 30')
parser.add_argument('--data-root', type=str, help='Path to root folder of the data.')

# model arguments
# S4D
parser.add_argument('--s4d-dims', type=int, default=1280, help='Num dimensions in S4D. Should not be changed, as it must match the output dimensions of Efficientnet. Default: 1280')
parser.add_argument('--s4d-states', type=int, default=1, help='Num of states per dimension in S4D. Default: 1')
parser.add_argument('--s4d-is-complex', action='store_true', help='Limit S4D to use real-numbers istead of complex.')
parser.add_argument('--s4d-lr', type=float, default=1e-3, help='Learning rate of S4D. Default: 1e-3')

# LSTM
parser.add_argument('--lstm-dims', type=int, default=1280, help='Num of states per dimension in S4D. Default: 1280')

# Readout
parser.add_argument('--readout-hidden-dims', type=int, default=64, help='Size of the hidden layer of the readout MLP. Default: 64')
parser.add_argument('--readout-no-bias', action='store_true', help='Do not use bias in readout.')

# YoloKP
parser.add_argument('--yolo-model-path', type=str, default="network.pt", help='Path to network file. (Default: network.py)')
parser.add_argument('--yolo-args-path', type=str, default="args.txt", help='Path to model arguments file. (Default: args.txt)')

args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
from model import model_registry 
import os
import torch
from torch import nn
from torch import optim
import time
import numpy as np
from dataloaders import dataset_registry 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from datetime import datetime


# Params
num_epochs = args.epochs 
print_interval = args.print_interval 
batch_size = args.batch_size
num_frames_per_sample = args.frames_per_sample
args.s4d_is_real = False if args.s4d_is_complex else True
args.readout_use_bias = False if args.readout_no_bias else True
args.train_backbone = False if args.no_train_backbone else True
if not args.train_backbone:
    print("WARNING, no backbone training might be faulty")
model_cls = model_registry[args.model] 
model_params = {"lstm_num_hidden": args.lstm_dims,
                "train_backbone": args.train_backbone,
                "num_readout_hidden": args.readout_hidden_dims,
                "readout_bias": args.readout_use_bias,
                "s4d_num_hidden": args.s4d_dims,
                "s4d_states": args.s4d_states,
                "s4d_is_real": args.s4d_is_real,
                "s4d_lr": args.s4d_lr, # Not used
                "yolo_model_path": args.yolo_model_path,
                "yolo_args_path": args.yolo_args_path,
                }


resolution = 448 if args.model == "YoloKP-S4D" else 224

# Dataset
init_dataloader = dataset_registry[args.dataset] 


train_dataloader = init_dataloader(partition="train",
                                   batch_size=batch_size,
                                   num_frames_per_sample=num_frames_per_sample,
                                   resolution=resolution,
                                   data_root=args.data_root) 
val_dataloader = init_dataloader(partition="val",
                                 batch_size=batch_size,
                                 num_frames_per_sample=num_frames_per_sample,
                                 resolution=resolution,
                                 data_root=args.data_root) 


num_classes = train_dataloader.dataset.num_classes
model = model_cls(num_classes=num_classes, **model_params).cuda()

if args.continue_training:
    print("Continue training.")
    checkpoint = torch.load(f'{args.model}-{args.dataset}.pth')
    model.load_state_dict(checkpoint, strict=False)
    model.train()



# Define your loss function
criterion = nn.CrossEntropyLoss()
# Define your optimizer

if args.lr != args.s4d_lr:
    print(f"Set {args.s4d_lr} lr for S4D")
    optimizer = optim.Adam([
                               {'params': model.efficientnet.parameters()},
                               {'params': model.readout.parameters()},
                               {'params': model.s4d.parameters(), 'lr': args.s4d_lr}
                           ], lr=args.lr)

else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

writer = SummaryWriter(f'runs/{args.dataset}/{args.model}/{datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
writer.add_hparams(model_params, {"lr": args.lr})

print("start training")

epoch_loss = []
val_loss = []
min_val_loss = np.inf
best_val_acc = 0
# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss.append(0)
    preds = []
    tgts = []

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        # Forward pass
        outputs = model(inputs.cuda())
        loss = criterion(outputs, targets.cuda().long())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
        preds += pred
        tgt = targets.cpu().numpy().tolist()
        tgts += tgt
        with torch.no_grad():
            if batch_idx % print_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item()}')
                print(f'pred {pred}, targets {tgt} acc {accuracy_score(tgts, preds)}')
        
        epoch_loss[-1] += loss.item()

    cm = ConfusionMatrixDisplay.from_predictions(tgts, preds)
    cm_norm = ConfusionMatrixDisplay.from_predictions(tgts, preds, normalize='true')
    train_acc = cm.confusion_matrix.diagonal() / cm.confusion_matrix.sum(axis=1)
    writer.add_scalar("Train Loss", epoch_loss[-1] / len(train_dataloader), epoch)
    writer.add_scalar("Train Accuracy", np.mean(train_acc), epoch)
    writer.add_figure("Train Confusion matrix", cm.figure_, epoch)
    writer.add_figure("Train Confusion matrix (normalized)", cm_norm.figure_, epoch)
    
    print(f"Avg loss on epoch {epoch}: {np.array(epoch_loss) / len(train_dataloader)}")


    model.eval()
    with torch.no_grad():
        val_loss.append(0)
        val_preds = []
        val_tgts = []

        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            # Forward pass
            outputs = model(inputs.cuda())
            loss = criterion(outputs, targets.cuda().long())

            pred = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
            val_preds += pred
            tgt = targets.cpu().numpy().tolist()
            val_tgts += tgt
            if batch_idx % print_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(val_dataloader)}], Loss: {loss.item()}')
                print(f'pred {pred}, targets {tgt} acc {accuracy_score(val_tgts, val_preds)}')
            
            val_loss[-1] += loss.item()

    cm = ConfusionMatrixDisplay.from_predictions(val_tgts, val_preds)
    cm_norm = ConfusionMatrixDisplay.from_predictions(val_tgts, val_preds, normalize='true')
    val_acc = cm.confusion_matrix.diagonal() / cm.confusion_matrix.sum(axis=1)

    writer.add_scalar("Val Loss", val_loss[-1] / len(val_dataloader), epoch)
    writer.add_scalar("Val Accuracy", np.mean(val_acc), epoch)
    writer.add_figure("Val Confusion matrix", cm.figure_, epoch)
    writer.add_figure("Val Confusion matrix (normalized)", cm_norm.figure_, epoch)

    
    # if val_loss[-1] < min_val_loss:  
    #     min_val_loss = val_loss[-1]    
    #     # Save the trained model (optional)
    #     torch.save(model.state_dict(), f'{args.model}.pth')

    if np.mean(val_acc) > best_val_acc:  
        best_val_acc = np.mean(val_acc)
        # Save the trained model (optional)
        torch.save(model.state_dict(), f'{args.model}-{args.dataset}.pth')

    
    print(f"Avg validation loss on epoch {epoch}: {np.array(val_loss) / len(val_dataloader)}")


