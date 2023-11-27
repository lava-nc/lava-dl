# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple, Union
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..yolo_base import YOLOBase
from .yolov3_ann import CNNBlock

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Network(YOLOBase):
    def __init__(self,
                 num_classes: int = 80,
                 anchors: List[List[Tuple[float, float]]] = [ 
                        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
                        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)] 
                    ]) -> None:
        super().__init__(num_classes=num_classes, anchors=anchors)
        
        self.layers_back_bone = nn.Sequential( OrderedDict([
            ('0_convbatch',     CNNBlock(3, 16, kernel_size = 3, stride = 1, padding = 1)),
            ('1_max',           nn.MaxPool2d(2, 2)),
            ('2_convbatch',     CNNBlock(16, 32, kernel_size = 3, stride = 1, padding = 1)),
            ('3_max',           nn.MaxPool2d(2, 2)),
            ('4_convbatch',     CNNBlock(32, 64, kernel_size = 3, stride = 1, padding = 1)),
            ('5_max',           nn.MaxPool2d(2, 2)),
            ('6_convbatch',     CNNBlock(64, 128, kernel_size = 3, stride = 1, padding = 1)),
            ('7_max',           nn.MaxPool2d(2, 2)),
            ('8_convbatch',     CNNBlock(128, 256, kernel_size = 3, stride = 1, padding = 1)),
            ('9_max',           nn.MaxPool2d(2, 2)),
            ('10_convbatch',    CNNBlock(256, 512, kernel_size = 3, stride = 1, padding = 1)),
            ('11_max',          MaxPoolStride1()),
            ('12_convbatch',    CNNBlock(512, 1024, kernel_size =  3, stride = 1, padding = 1)),
            ('13_convbatch',    CNNBlock(1024, 256, kernel_size =  1, stride = 1, padding = 0)),
        ]))
        
        self.yolo_0_pre = nn.Sequential(OrderedDict([
            ('14_convbatch',    CNNBlock(256, 512, kernel_size = 3, stride = 1, padding = 1)),
            ('15_conv',         nn.Conv2d(512, self.num_output, 1, 1, 0)),
        ]))

        self.up_1 = nn.Sequential(OrderedDict([
            ('17_convbatch',    CNNBlock(256, 128, kernel_size = 1, stride = 1, padding = 0)),
            ('18_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
        ]))

        self.yolo_1_pre = nn.Sequential(OrderedDict([
            ('19_convbatch',    CNNBlock(128 + 256, 256, kernel_size = 3, stride = 1, padding = 1)),
            ('20_conv',         nn.Conv2d(256, self.num_output, 1, 1, 0)),
        ]))
        
        # standard imagenet normalization of RGB images
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        self.normalize_std  = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])

    def forward(
        self,
        input: torch.tensor,
    ) -> Tuple[Union[torch.tensor, List[torch.tensor]], torch.tensor]:
        
        if self.normalize_mean.device != input.device:
            self.normalize_mean = self.normalize_mean.to(input.device)
            self.normalize_std = self.normalize_std.to(input.device)
            
        # we remove time dependency 
        input = torch.reshape(input, (input.shape[0], input.shape[1], input.shape[2], input.shape[3]))
        input = (input - self.normalize_mean) / self.normalize_std
        
        outputs = [] 
        x_b_0 = self.layers_back_bone[:9](input)
        x_b_full = self.layers_back_bone[9:](x_b_0)
        y0 = self.yolo_0_pre(x_b_full)
        x_up = self.up_1(x_b_full)
        x_up = torch.cat((x_up, x_b_0), 1)
        y1 = self.yolo_1_pre(x_up)
        
        y0 = y0.view(y0.size(0), self.num_anchors, self.num_classes + 5, y0.size(2), y0.size(3)) 
        y0 = y0.permute(0, 1, 3, 4, 2)
        
        y1 = y1.view(y1.size(0), self.num_anchors, self.num_classes + 5, y1.size(2), y1.size(3)) 
        y1 = y1.permute(0, 1, 3, 4, 2) 
        
        outputs = [y0, y1]
     
        if not self.training:
            outputs = [item[..., None] for item in outputs]
            outputs = torch.concat([self.yolo(p, a) for (p, a)
                                            in zip(outputs,  self.anchors)],
                                           dim=1)
        return outputs, []