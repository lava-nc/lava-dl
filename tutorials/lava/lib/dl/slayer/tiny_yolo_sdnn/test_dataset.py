# nosec # noqa
import os
import argparse
from typing import Any, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=[0], help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',   type=int, default=1,  help='batch size for dataloader')
    parser.add_argument('-verbose', default=False, action='store_true', help='lots of debug printouts')
    # Model
    parser.add_argument('-model', type=str, default='tiny_yolov3_str_events', help='network model')
    # Sparsity
    parser.add_argument('-sparsity', action='store_true', default=False, help='enable sparsity loss')
    parser.add_argument('-sp_lam',   type=float, default=0.01, help='sparsity loss mixture ratio')
    parser.add_argument('-sp_rate',  type=float, default=0.01, help='minimum rate for sparsity penalization')
    # Optimizer
    parser.add_argument('-lr',  type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-wd',  type=float, default=1e-5,   help='optimizer weight decay')
    parser.add_argument('-lrf', type=float, default=0.01,   help='learning rate reduction factor for lr scheduler')
    # Network/SDNN
    parser.add_argument('-threshold',  type=float, default=0.1, help='neuron threshold')
    parser.add_argument('-tau_grad',   type=float, default=0.1, help='surrogate gradient time constant')
    parser.add_argument('-scale_grad', type=float, default=0.2, help='surrogate gradient scale')
    parser.add_argument('-clip',       type=float, default=10, help='gradient clipping limit')
    # Pretrained model
    parser.add_argument('-load', type=str, default='', help='pretrained model')
    # Target generation
    parser.add_argument('-tgt_iou_thr', type=float, default=0.5, help='ignore iou threshold in target generation')
    # YOLO loss
    parser.add_argument('-lambda_coord',    type=float, default=1.0, help='YOLO coordinate loss lambda')
    parser.add_argument('-lambda_noobj',    type=float, default=2.0, help='YOLO no-object loss lambda')
    parser.add_argument('-lambda_obj',      type=float, default=2.0, help='YOLO object loss lambda')
    parser.add_argument('-lambda_cls',      type=float, default=4.0, help='YOLO class loss lambda')
    parser.add_argument('-lambda_iou',      type=float, default=2.0, help='YOLO iou loss lambda')
    parser.add_argument('-alpha_iou',       type=float, default=0.8, help='YOLO loss object target iou mixture factor')
    parser.add_argument('-label_smoothing', type=float, default=0.1, help='YOLO class cross entropy label smoothing')
    parser.add_argument('-track_iter',      type=int,  default=1000, help='YOLO loss tracking interval')
    # Experiment
    parser.add_argument('-exp',  type=str, default='',   help='experiment differentiater string')
    parser.add_argument('-seed', type=int, default=None, help='random seed of the experiment')
    # Training
    parser.add_argument('-epoch',  type=int, default=200, help='number of epochs to run')
    parser.add_argument('-warmup', type=int, default=10,  help='number of epochs to warmup')
    # dataset
    parser.add_argument('-dataset',     type=str,   default='PropheseeAutomotive', help='dataset to use [BDD100K, PropheseeAutomotive]')
    parser.add_argument('-path',        type=str,   default='/home/lecampos/data/prophesee', help='dataset path')
    parser.add_argument('-output_dir',  type=str,   default='.', help='directory in which to put log folders')
    parser.add_argument('-num_workers', type=int,   default=0, help='number of dataloader workers')
    parser.add_argument('-aug_prob',    type=float, default=0.2, help='training augmentation probability')
    parser.add_argument('-clamp_max',   type=float, default=5.0, help='exponential clamp in height/width calculation')

    args = parser.parse_args()
    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    print('Creating Dataset')
    train_set = obd.dataset.PropheseeAutomotive(root=args.path, train=True, 
                                                augment_prob=args.aug_prob, 
                                                randomize_seq=True, 
                                                seq_len=100)
    test_set = obd.dataset.PropheseeAutomotive(root=args.path, train=False,
                                                randomize_seq=True, 
                                                seq_len=100)
    
    train_loader = DataLoader(train_set,
                                batch_size=args.b,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)
    test_loader = DataLoader(test_set,
                                batch_size=args.b,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)        
   

    print('Training/Testing Loop')
    for epoch in range(args.epoch):
        print(f'{epoch=}')
        print('Training Loop')
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
        print('Testing Loop')
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
           
