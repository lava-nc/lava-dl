import torch
import h5py
import torch.nn.functional as F
from typing import List, Tuple, Callable
from lava.lib.dl import slayer
import numpy as np
import matplotlib.pyplot as plt
import sys
path = ['/home/dbendaya/work/ContinualLearning/tinyYolov3_lava/YOLOsdnn/']
sys.path.extend(path)
from yolo_base import YOLOBase

from object_detection.dataset.utils import storeData
from .model_utils import quantize_8bit, quantize_5bit, event_rate, SparsityMonitor

#### sdnn_single_head_KP_combination_yolo model

class Network(YOLOBase):
    def __init__(self, args,
                 num_classes=80,
                 anchors=[ [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)] ],
                 clamp_max=5.0):

        anchors = anchors*args.combined_models
        super().__init__(num_classes=args.num_classes, anchors=anchors, clamp_max=args.clamp_max)
        self.device = args.gpu[0]
        # self.training = args.train

        print(f'loading {args.combined_models} heads:')
        for k,h in enumerate(args.model):
            print(k,h)
            exec('from models.sdnn_%s import Network as Head%d'%(h,k))
            exec('self.Head%d = Head%d(threshold=args.threshold, tau_grad=args.tau_grad, scale_grad=args.scale_grad, num_classes=args.num_classes, clamp_max=args.clamp_max).to(self.device)'%(k,k))
            self.anchors[k,:,:] = eval(f'self.Head{k}.anchors')

        
    def forward(self, input, sparsity_monitor: SparsityMonitor=None):        
        output, count = [], []
        # print('flag train:',self.training)
        for name, head in self.named_children():
            out, cnt = head(input, sparsity_monitor)
            # storeData.save([out, cnt],name+'.pkl')
            output += out[0] if self.training else out
            count  += cnt            
        if not self.training:
            output = torch.concat(output, dim=1)
            output = output.unsqueeze(0).view(input.shape[0], -1, self.num_classes+5, output.shape[-1])
            count = [torch.stack(count, dim=1).mean(1)]
        return output, count
            
    def grad_flow(self, path):
        # helps monitor the gradient flow
        head_grad_flow = []
        for h in self.children():
            head_grad_flow += h.gradflow(path)      
        
        plt.figure()
        plt.semilogy(head_grad_flow)
        plt.savefig(path + 'HeadGradFlow.png')
        plt.close()

        return head_grad_flow

    def load_model(self, model_file: str):
        for k, head in enumerate(self.children()):
            head.load_model(model_file[k])
            
    ############ EXTENSIONS FROM YOLOBase ##################        

    def validate_gradients(self) -> None:
        valid_gradients = True
        for head in self.children():       
            for name, param in head.named_parameters():
                if param.grad is not None:
                    valid_gradients = not (torch.isnan(param.grad).any()
                                           or torch.isinf(param.grad).any())
                    if not valid_gradients:
                        break
            if not valid_gradients:
                head.zero_grad()

