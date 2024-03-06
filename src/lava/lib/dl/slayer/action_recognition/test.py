import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model to train on. See model registry in model.py for a list of available models (efficientnet-b0-LSTM, efficientnet-b0-S4D, ..).')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size. Default: 8')
# parser.add_argument('--epochs', type=int, default=100, help='Num epochs. Default: 100')
parser.add_argument('--print-interval', type=int, default=100, help='Print information each N batches. Default: 100')
# parser.add_argument('--lr', type=float, metavar="LEARNING_RATE", default=1e-3, help='Learning rate. Default: 1e-3')

# Dataset
parser.add_argument('--dataset', type=str, default="NTU", help='NTU or HARDVS. Default: NTU')
parser.add_argument('--frames-per-sample', type=int, default=30, help='Num frames per sample. Default: 30')
parser.add_argument('--data-root', type=str, help='Path to root folder of the data.')

# model arguments
parser.add_argument('--s4d-dims', type=int, default=1280, help='Num dimensions in S4D. Should not be changed, as it must match the output dimensions of Efficientnet. Default: 1280')
parser.add_argument('--s4d-states', type=int, default=1, help='Num of states per dimension in S4D. Default: 1')
parser.add_argument('--s4d-is-complex', action='store_true', help='Limit S4D to use real-numbers istead of complex.')
# parser.add_argument('--s4d-lr', type=float, default=1e-3, help='Learning rate of S4D. Default: 1e-3')
parser.add_argument('--lstm-dims', type=int, default=1280, help='Num of states per dimension in S4D. Default: 1280')
parser.add_argument('--readout-hidden-dims', type=int, default=64, help='Size of the hidden layer of the readout MLP. Default: 64')
parser.add_argument('--readout-no-bias', action='store_true', help='Do not use bias in readout.')

# YoloKP
parser.add_argument('--yolo-model-path', type=str, default="network.pt", help='Path to network file. (Default: network.py)')
parser.add_argument('--yolo-args-path', type=str, default="args.txt", help='Path to model arguments file. (Default: args.txt)')
args = parser.parse_args()

from model import model_registry 
import os
from torchvision import transforms
import torch
from torch import nn
from torch import optim
import time
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from dataloaders import dataset_registry 

# Params
learning_rate = 0.0 
print_interval = args.print_interval 
batch_size = args.batch_size
num_frames_per_sample = args.frames_per_sample
args.s4d_is_real = False if args.s4d_is_complex else True
args.readout_use_bias = False if args.readout_no_bias else True
model_cls = model_registry[args.model] 
model_params = {"lstm_num_hidden": args.lstm_dims,
                "num_readout_hidden": args.readout_hidden_dims,
                "readout_bias": args.readout_use_bias,
                "s4d_num_hidden": args.s4d_dims,
                "s4d_states": args.s4d_states,
                "s4d_is_real": args.s4d_is_real,
                "s4d_lr": 0.0,
                "yolo_model_path": args.yolo_model_path,
                "yolo_args_path": args.yolo_args_path,
                }

resolution = 448 if args.model == "YoloKP-S4D" else 224

# Dataset
init_dataloader = dataset_registry[args.dataset] 

test_dataloader = init_dataloader(partition="test",
                                  batch_size=batch_size,
                                  num_frames_per_sample=num_frames_per_sample,
                                  resolution=resolution,
                                  data_root=args.data_root) 

num_classes = test_dataloader.dataset.num_classes
model = model_cls(num_classes=num_classes, **model_params).cuda()

checkpoint = torch.load(f"{args.model}-{args.dataset}.pth")
model.load_state_dict(checkpoint)
model.eval()

print("start testing")
with torch.no_grad():
    test_preds = []
    test_tgts = []

    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        # Forward pass
        outputs = model(inputs.cuda())

        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
        test_preds += pred
        tgt = targets.cpu().numpy().tolist()
        test_tgts += tgt
        if batch_idx % print_interval == 0:
            print(f'Batch [{batch_idx+1}/{len(test_dataloader)}]')
            print(f'pred {pred}, targets {tgt} acc {accuracy_score(test_tgts, test_preds)}')


    cm = ConfusionMatrixDisplay.from_predictions(test_tgts, test_preds)
    cm_norm = ConfusionMatrixDisplay.from_predictions(test_tgts, test_preds, normalize='true')
    test_acc = cm.confusion_matrix.diagonal() / cm.confusion_matrix.sum(axis=1)
    print(f'Test Acc {test_acc}, mean {np.mean(test_acc)}')
    print(cm.confusion_matrix)
    print(cm_norm.confusion_matrix)



