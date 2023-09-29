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
from object_detection.dataset.coco import COCO
from object_detection.dataset.bdd100k import BDD
from object_detection.dataset.utils import collate_fn
from yolo_base import YOLOBase, YOLOLoss, YOLOtarget
from object_detection.boundingbox.metrics import APstats
from object_detection.boundingbox.utils import (
    Height, Width, annotation_from_tensor, mark_bounding_boxes,
)
from object_detection.boundingbox.utils import non_maximum_suppression as nms


BOX_COLOR_MAP = [(np.random.randint(256),
                  np.random.randint(256),
                  np.random.randint(256))
                 for i in range(80)]


class COCO128(COCO):
    def __init__(self,
                 root: str = './',
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 augment_prob: float = 0.0) -> None:
        super().__init__(root, size, train, augment_prob)

    def __len__(self):
        return 128


def quantize_8bit(x, scale = (1 << 6), descale=False):
    if descale is False:
        return slayer.utils.quantize(x, step=2 / scale).clamp(-256 / scale, 254 / scale)
    else:
        return slayer.utils.quantize(x, step=2 / scale).clamp(-256 / scale, 254 / scale) * scale


def event_rate(x):
    if x.shape[-1] == 1:
        return torch.mean(torch.abs((x) > 0).to(x.dtype)).item()
    else:
        return torch.mean(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item()

class SparsityMonitor:
    def __init__(self, max_rate=0.01, lam=1):
        self.max_rate = max_rate
        self.lam = lam
        self.loss_list = []

    def clear(self):
        self.loss_list = []

    @property
    def loss(self):
        return self.lam * sum(self.loss_list)

    def append(self, x):
        mean_event_rate = torch.mean(torch.abs(x))
        self.loss_list.append(F.mse_loss(F.relu(mean_event_rate - self.max_rate),
                                         torch.zeros_like(mean_event_rate)))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=[0], help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b'  , type=int, default=32,  help='batch size for dataloader')
    # Sparsity
    parser.add_argument('-sparsity', default=False, action='store_true', help='enable sparsity loss')
    parser.add_argument('-sp_lam', type=float, default=0.01, help='sparsity loss mixture ratio')
    parser.add_argument('-sp_rate', type=float, default=0.01, help='sparsity penalization rate')
    # Optimizer
    parser.add_argument('-lr' , type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-wd' , type=float, default=1e-5 , help='optimizer weight decay')
    parser.add_argument('-lrf', type=float, default=0.001, help='learning rate factor for lr scheduler')
    # Network/SDNN
    parser.add_argument('-threshold' , type=float, default=0.1, help='neuron threshold')
    parser.add_argument('-tau_grad'  , type=float, default=0.1, help='surrogate gradient time constant')
    parser.add_argument('-scale_grad', type=float, default=0.2, help='surrogate gradient scale')
    parser.add_argument('-clip'      , type=float, default=10,  help='gradient clipping limit')
    # Pretrained model
    parser.add_argument('-load'      , type=str, default='',  help='pretrained model')
    # Target generation
    parser.add_argument('-tgt_iou_thr', type=float, default=0.5, help='ignore iou threshold in target generation')
    # YOLO loss
    parser.add_argument('-lambda_coord'   , type=float, default= 1.0, help='YOLO coordinate loss lambda')
    parser.add_argument('-lambda_noobj'   , type=float, default=10.0, help='YOLO no-object loss lambda')
    parser.add_argument('-lambda_obj'     , type=float, default= 5.0, help='YOLO object loss lambda')
    parser.add_argument('-lambda_cls'     , type=float, default= 1.0, help='YOLO class loss lambda')
    parser.add_argument('-lambda_iou'     , type=float, default= 1.0, help='YOLO iou loss lambda')
    parser.add_argument('-alpha_iou'      , type=float, default=0.25, help='YOLO loss object target iou mixture factor')
    parser.add_argument('-label_smoothing', type=float, default=0.10, help='YOLO class cross entropy label smoothing')
    parser.add_argument('-track_iter'     , type=int,   default=1000, help='YOLO loss tracking interval')
    # Experiment
    parser.add_argument('-exp', type=str, default='', help='experiment differentiater string')
    parser.add_argument('-seed', type=int, default=None, help='random seed of the experiment')
    # Training
    parser.add_argument('-epoch' , type=int, default=50, help='number of epochs to run')
    parser.add_argument('-warmup',type=int, default= 10, help='number of epochs to warmup')
    # dataset
    parser.add_argument('-dataset', type=str, default='BDD100K', help='dataset to use [COCO, BDD100K, PASCAL]')
    parser.add_argument('-path', type=str, default='/data-raid/sshresth/data/bdd100k/MOT2020/bdd100k', help='dataset path')
    parser.add_argument('-aug_prob', type=float, default=0.2, help='training augmentation probability')
    parser.add_argument('-subset', default=False, action='store_true', help='use COCO128 subset')
    parser.add_argument('-output_dir', type=str,   default=".",    help="directory in which to put log folders")
    parser.add_argument('-num_workers', type=int,   default=12,    help="number of dataloader workers")
    parser.add_argument('-clamp_max', type=float,   default=5.0,    help="exponential clamp in height/width calculation")
    parser.add_argument('-verbose', default=False, action='store_true', help='lots of debug printouts')

    args = parser.parse_args()
    
    identifier = 'SDNN_' + args.exp if len(args.exp) > 0 else 'SDNN'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}'.format(args.seed)

    trained_folder = args.output_dir + '/Trained_' + identifier if len(identifier) > 0 else args.output_dir + '/Trained'
    logs_folder = args.output_dir + '/Logs_' + identifier if len(identifier) > 0 else args.output_dir + '/Logs'

    print(trained_folder)
    writer = SummaryWriter(args.output_dir + '/runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    classes_output = {'COCO': 80, 'BDD100K': 11}

    print('making net') 
    if len(args.gpu) == 1:
        net = Network(threshold=args.threshold,
                      tau_grad=args.tau_grad,
                      scale_grad=args.scale_grad,
                      num_classes=classes_output[args.dataset],
                      clamp_max=args.clamp_max).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(threshold=args.threshold,
                                            tau_grad=args.tau_grad,
                                            scale_grad=args.scale_grad,
                                            num_classes=classes_output[args.dataset],
                                            clamp_max=args.clamp_max).to(device),
                                    device_ids=args.gpu)
        module = net.module

    if args.sparsity:
        sparsity_montior = SparsityMonitor(max_rate=args.sp_rate, lam=args.sp_lam)
    else:
        sparsity_montior = None

    print('loading net') 
    if args.load != '':
        print(f'Initializing model from {args.load}')
        module.load_model(args.load)

    print('module.init_model') 
    module.init_model((448, 448))

    # Define optimizer module.
    # optimizer = torch.optim.RAdam(net.parameters(),
    #                               lr=args.lr,
    #                               weight_decay=args.wd)
    print("optimizer") 
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)
    # optimizer = torch.optim.SGD(net.parameters(),
    #                             lr=args.lr,
    #                             weight_decay=args.wd)
    
    # Define learning rate scheduler
    # lf = lambda x: (1 - x / (args.epoch - 1)) * (1 - args.lrf) + args.lrf
    lf = lambda x: min(x / args.warmup, 1) * ((1 + np.cos(x * np.pi / args.epoch)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    yolo_target = YOLOtarget(anchors=net.anchors,
                             scales=net.scale,
                             num_classes=net.num_classes,
                             ignore_iou_thres=args.tgt_iou_thr)

    print('dataset') 
    if args.dataset == 'COCO':
        if args.subset:
            train_set = COCO128(root=args.path, train=True,
                                augment_prob=args.aug_prob)
            test_set = COCO128(root=args.path, train=False)
            train_loader = DataLoader(train_set,
                                    batch_size=args.b,
                                    shuffle=False,
                                    collate_fn=yolo_target.collate_fn,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
            test_loader = DataLoader(train_set,  # test_set,
                                    batch_size=args.b,
                                    shuffle=False,
                                    collate_fn=yolo_target.collate_fn,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
        else:
            train_set = COCO(root=args.path, train=True, augment_prob=args.aug_prob)
            test_set = COCO(root=args.path, train=False)
            train_loader = DataLoader(train_set,
                                    batch_size=args.b,
                                    shuffle=True,
                                    collate_fn=yolo_target.collate_fn,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
            test_loader = DataLoader(test_set,
                                    batch_size=args.b,
                                    shuffle=True,
                                    collate_fn=yolo_target.collate_fn,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
    elif args.dataset == 'BDD100K':
            train_set = BDD(root=args.path, dataset='track', train=True, augment_prob=args.aug_prob, randomize_seq=True)
            test_set = BDD(root=args.path, dataset='track', train=False, randomize_seq=True)
            train_loader = DataLoader(train_set,
                                    batch_size=args.b,
                                    shuffle=True,
                                    collate_fn=yolo_target.collate_fn,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
            test_loader = DataLoader(test_set,
                                    batch_size=args.b,
                                    shuffle=True,
                                    collate_fn=yolo_target.collate_fn,
                                    num_workers=args.num_workers,
                                    pin_memory=True)


    print('yolo_loss') 
    yolo_loss = YOLOLoss(anchors=net.anchors,
                         lambda_coord=args.lambda_coord,
                         lambda_noobj=args.lambda_noobj,
                         lambda_obj=args.lambda_obj,
                         lambda_cls=args.lambda_cls,
                         lambda_iou=args.lambda_iou,
                         alpha_iou=args.alpha_iou,
                         label_smoothing=args.label_smoothing).to(device)

    print('stats') 
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')

    print('loss_tracker') 
    loss_tracker = dict(coord=[], obj=[], noobj=[], cls=[], iou=[])
    loss_order = ['coord', 'obj', 'noobj', 'cls', 'iou']

    print('train loop') 

    for epoch in range(args.epoch):
        t_st = datetime.now()
        ap_stats = APstats(iou_threshold=0.5)

        print(f'{epoch=}')
        for i, (inputs, targets, bboxes) in enumerate(train_loader):

            print(f'{i=}')  if args.verbose else None

            net.train()
            print('inputs') if args.verbose else None
            inputs = inputs.to(device)

            print('forward') if args.verbose else None
            predictions, counts = net(inputs, sparsity_montior)

            loss, loss_distr = yolo_loss(predictions, targets)
            if sparsity_montior is not None:
                loss += sparsity_montior.loss
                sparsity_montior.clear()
            
            if torch.isnan(loss):
                print("loss is nan, continuing")
                continue
            optimizer.zero_grad()
            loss.backward()
            net.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            if i < 10:
                net.grad_flow(path=trained_folder + '/')

            # MAP calculations
            T = inputs.shape[-1]
            try:
                # predictions = torch.concat([net.yolo(predictions[0], net.anchors[0]),
                #                             net.yolo(predictions[1], net.anchors[1])], dim=1)
                predictions = torch.concat([net.yolo(p, a) for (p, a)
                                            in zip(predictions, net.anchors)],
                                            dim=1)
            except AssertionError:
                print("assertion error on MAP predictions calculation train set. continuing")
                continue
            predictions = [nms(predictions[..., t]) for t in range(T)]

            for t in range(T):
                ap_stats.update(predictions[t], bboxes[t])

            if not torch.isnan(loss):
                stats.training.loss_sum += loss.item() * inputs.shape[0]
            stats.training.num_samples += inputs.shape[0]
            stats.training.correct_samples = ap_stats[:] * stats.training.num_samples

            processed = i * train_loader.batch_size
            total = len(train_loader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
            header_list = [f'Train: [{processed}/{total} '
                        f'({100.0 * processed / total:.0f}%)]']
            header_list += ['Event Rate: ['
                            + ', '.join([f'{c.item():.2f}'
                                            for c in counts[0]]) + ']']
            header_list += [f'Coord loss: {loss_distr[0].item()}']
            header_list += [f'Obj   loss: {loss_distr[1].item()}']
            header_list += [f'NoObj loss: {loss_distr[2].item()}']
            header_list += [f'Class loss: {loss_distr[3].item()}']
            header_list += [f'IOU   loss: {loss_distr[4].item()}']
            
            if i % args.track_iter == 0:
                plt.figure()
                for loss_idx, loss_key in enumerate(loss_order):
                    loss_tracker[loss_key].append(loss_distr[loss_idx].item())
                    plt.semilogy(loss_tracker[loss_key], label=loss_key)
                    if not args.subset:
                        writer.add_scalar(f'Loss Tracker/{loss_key}',
                                        loss_distr[loss_idx].item(),
                                        len(loss_tracker[loss_key]) - 1)
                plt.xlabel(f'iters (x {args.track_iter})')
                plt.legend()
                plt.savefig(f'{trained_folder}/yolo_loss_tracker.png')
                plt.close()
            stats.print(epoch, i, samples_sec, header=header_list)

        t_st = datetime.now()
        ap_stats = APstats(iou_threshold=0.5)
        for i, (inputs, targets, bboxes) in enumerate(test_loader):
            net.eval()

            with torch.no_grad():
                inputs = inputs.to(device)
                predictions, counts = net(inputs)

                T = inputs.shape[-1]
                predictions = [nms(predictions[..., t]) for t in range(T)]
                for t in range(T):
                    ap_stats.update(predictions[t], bboxes[t])

                stats.testing.loss_sum += loss.item() * inputs.shape[0]
                stats.testing.num_samples += inputs.shape[0]
                stats.testing.correct_samples = ap_stats[:] * stats.testing.num_samples

                processed = i * test_loader.batch_size
                total = len(test_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / test_loader.batch_size
                header_list = [f'Test: [{processed}/{total} '
                               f'({100.0 * processed / total:.0f}%)]']
                header_list += ['Event Rate: ['
                                + ', '.join([f'{c.item():.2f}'
                                                for c in counts[0]]) + ']']
                header_list += [f'Coord loss: {loss_distr[0].item()}']
                header_list += [f'Obj   loss: {loss_distr[1].item()}']
                header_list += [f'NoObj loss: {loss_distr[2].item()}']
                header_list += [f'Class loss: {loss_distr[3].item()}']
                header_list += [f'IOU   loss: {loss_distr[4].item()}']
                stats.print(epoch, i, samples_sec, header=header_list)

        if not args.subset:
            writer.add_scalar('Loss/train', stats.training.loss, epoch)
            writer.add_scalar('mAP@50/train', stats.training.accuracy, epoch)
            writer.add_scalar('mAP@50/test', stats.testing.accuracy, epoch)

        stats.update()
        stats.plot(path=trained_folder + '/')
        b = -1
        image = Image.fromarray(np.uint8(
            inputs[b, :, :, :, 0].cpu().data.numpy().transpose([1, 2, 0]) * 255
        ))
        annotation = annotation_from_tensor(
            predictions[0][b],
            {'height': image.height, 'width': image.width},
            test_set.classes,
            confidence_th=0
        )
        marked_img = mark_bounding_boxes(
            image, annotation['annotation']['object'],
            box_color_map=BOX_COLOR_MAP, thickness=5
        )

        image = Image.fromarray(np.uint8(
            inputs[b, :, :, :, 0].cpu().data.numpy().transpose([1, 2, 0]) * 255
        ))
        annotation = annotation_from_tensor(
            bboxes[0][b],
            {'height': image.height, 'width': image.width},
            test_set.classes,
            confidence_th=0
        )
        marked_gt = mark_bounding_boxes(
            image, annotation['annotation']['object'],
            box_color_map=BOX_COLOR_MAP, thickness=5
        )

        marked_images = Image.new('RGB', (marked_img.width + marked_gt.width, marked_img.height))
        marked_images.paste(marked_img, (0, 0))
        marked_images.paste(marked_gt, (marked_img.width, 0))
        if not args.subset:
            writer.add_image('Prediction', transforms.PILToTensor()(marked_images), epoch)

        if stats.testing.best_accuracy is True:
            torch.save(module.state_dict(), trained_folder + '/network.pt')
            if inputs.shape[-1] == 1:
                marked_images.save(f'{trained_folder}/prediction_{epoch}_{b}.jpg')
            else:
                video_dims = (2 * marked_img.width, marked_img.height)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
                video = cv2.VideoWriter(f'{trained_folder}/prediction_{epoch}_{b}.mp4',
                                        fourcc, 10, video_dims)
                for t in range(inputs.shape[-1]):
                    image = Image.fromarray(np.uint8(inputs[b, :, :, :, t].cpu().data.numpy().transpose([1, 2, 0]) * 255))
                    annotation = annotation_from_tensor(predictions[t][b],
                                                        {'height': image.height,
                                                         'width': image.width},
                                                        test_set.classes,
                                                        confidence_th=0)
                    marked_img = mark_bounding_boxes(image, annotation['annotation']['object'],
                                                     box_color_map=BOX_COLOR_MAP, thickness=5)
                    image = Image.fromarray(np.uint8(inputs[b, :, :, :, t].cpu().data.numpy().transpose([1, 2, 0]) * 255))
                    annotation = annotation_from_tensor(bboxes[t][b],
                                                        {'height': image.height,
                                                         'width': image.width},
                                                        test_set.classes,
                                                        confidence_th=0)
                    marked_gt = mark_bounding_boxes(image, annotation['annotation']['object'],
                                                    box_color_map=BOX_COLOR_MAP, thickness=5)
                    marked_images = Image.new('RGB', (marked_img.width + marked_gt.width, marked_img.height))
                    marked_images.paste(marked_img, (0, 0))
                    marked_images.paste(marked_gt, (marked_img.width, 0))
                    video.write(cv2.cvtColor(np.array(marked_images), cv2.COLOR_RGB2BGR))
                video.release()

        stats.save(trained_folder + '/')

    if hasattr(module, 'export_hdf5'):
        module.load_state_dict(torch.load(trained_folder + '/network.pt'))
        module.export_hdf5(trained_folder + '/network.net')

    if not args.subset:
        params_dict = {}
        for key, val in args._get_kwargs():
            params_dict[key] = str(val)
        writer.add_hparams(params_dict, {'mAP@50': stats.testing.max_accuracy})
        writer.flush()
        writer.close()