
import torch
from torch import nn, Tensor
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn import MSELoss, CrossEntropyLoss
from torchvision import transforms
from typing import Iterable, Callable, Dict, List
from PIL import Image
from lava.lib.dl.slayer.image_classification.imagenet_dataset import ImageNet

from torchvision.models import efficientnet_b0, EfficientNet
from lava.lib.dl.slayer.image_classification.efficientnet import my_efficientnet_b0, MyEfficientNet

from lava.lib.dl.slayer.image_classification.piecewise_linear_silu import PiecewiseLinearSiLU
from lava.lib.dl.slayer.image_classification.relu1 import ReLU1 
from lava.lib.dl.slayer.knowledge_distillation.knowledge_distillation import FeatureWiseKnowledgeDist, KnowledgeDist

from lava.lib.dl.slayer.action_recognition.model import EfficientNetS4D, SlayerCNN, PlSiLUEfficientNetS4D
import numpy as np

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, split='train'):
        super().__init__()
        with open(file_list, "r") as f:
            self.fns = f.readlines()

        resolution = 224 
        self.preprocess = transforms.Compose([
                transforms.Resize(resolution + 2, antialias=None),  # image batch, resize smaller edge to 256
                transforms.CenterCrop(resolution),  # image batch, center crop to square 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def _load_image(self, path: str) -> Image.Image:

        try:
            return Image.open(path[:-1]).convert('RGB')
        except:
            print(path, path[:-1])

    def __getitem__(self, idx):
        inp = self.preprocess(self._load_image(self.fns[idx]))
        tgt = torch.zeros_like(inp)
        return inp, tgt
    
    def __len__(self):
        return len(self.fns)


def train_original_to_pl_silu():
    print("Load model")
    teacher = efficientnet_b0(weights='IMAGENET1K_V1').cuda()
    teacher.eval()
    
    student = my_efficientnet_b0(weights='IMAGENET1K_V1',
                                 activation=PiecewiseLinearSiLU,
                                 scale_act=ReLU1).cuda()
    print("done.")

    print("Create dataset")
    imagenet_dataset = ImageNetDataset(file_list="/nas-data/pweidel/datasets/imagenet/index_file_train.txt", split='train')
    dataloader = torch.utils.data.DataLoader(dataset=imagenet_dataset,
                                             batch_size=32,
                                             num_workers=32,
                                             pin_memory=True,
                                             shuffle=True)
    print("done.")
    
    print("Create KD") 
    fwkd = FeatureWiseKnowledgeDist(teacher, student, teacher.features, student.features, dataloader, lr=1e-3)
    print("done.")

    print("Start training")
    fwkd.train()
    print("done.")

    print("Save network")
    fwkd.save_student("EffSiLU.pt")
    print("done.")
    
def train_original_to_slayer_cnn():
    print("Create dataset")

    batch_size = 256 
    resolution = 224 
    preprocess = transforms.Compose([
            transforms.Resize(resolution + 2, antialias=None),  # image batch, resize smaller edge to 256
            transforms.CenterCrop(resolution),  # image batch, center crop to square 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    train_imagenet_dataset = ImageNet(root="/nas-data/pweidel/datasets/imagenet", split='train', transform=preprocess)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_imagenet_dataset,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             pin_memory=True,
                                             shuffle=True)
    val_imagenet_dataset = ImageNet(root="/nas-data/pweidel/datasets/imagenet", split='val', transform=preprocess)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_imagenet_dataset,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             pin_memory=True,
                                             shuffle=True)


    print("Load model")
    teacher = efficientnet_b0(weights='IMAGENET1K_V1').cuda()
    teacher.eval()

    student_model_params = {
                            "readout_bias": True, 
                            }
    
    student = SlayerCNN(num_classes=1000, **student_model_params).cuda()
    print("done.")

    print("Create KD") 
    kd = KnowledgeDist(teacher,
                       student,
                       student.parameters(),
                       train_dataloader,
                       val_dataloader,
                       lr=1e-3,
                       n_epochs=100,
                       print_interval=100,)
    print("done.")

    print("Start training")
    kd.train()
    print("done.")

    print("Save network")
    kd.save_student("SlayerCNN_pretrained.pt")
    print("done.")

def train_pl_silu_to_no_scale():

    teacher = my_efficientnet_b0(weights='IMAGENET1K_V1',
                                 activation=PiecewiseLinearSiLU,
                                 scale_act=ReLU1).cuda()
    checkpoint = torch.load("EffSiLU.pt")
    teacher.load_state_dict(checkpoint)
    teacher.eval()
    student = my_efficientnet_b0(weights='IMAGENET1K_V1',
                                 activation=PiecewiseLinearSiLU,
                                 scale=False).cuda()
    

    
    imagenet_dataset = ImageNetDataset(file_list="/nas-data/pweidel/datasets/imagenet/index_file_train.txt", split='train')
    dataloader = torch.utils.data.DataLoader(dataset=imagenet_dataset,
                                             batch_size=32,
                                             num_workers=32,
                                             pin_memory=True,
                                             shuffle=True)
    
    fwkd = FeatureWiseKnowledgeDist(teacher, student, teacher.features, student.features, dataloader, lr=1e-3)
    fwkd.train(device='cuda')
    fwkd.save_student("EffSiLUNoScale.pt")
 


def train_no_scale_classifier():

    teacher = my_efficientnet_b0(weights='IMAGENET1K_V1').cuda()
    teacher.eval()
    student = my_efficientnet_b0(weights='IMAGENET1K_V1',
                                 activation=PiecewiseLinearSiLU,
                                 scale=False).cuda()

    checkpoint = torch.load("EffSiLUNoScale.pt")
    student.load_state_dict(checkpoint)
    
    imagenet_dataset = ImageNetDataset(file_list="/nas-data/pweidel/datasets/imagenet/index_file_train.txt", split='train')
    dataloader = torch.utils.data.DataLoader(dataset=imagenet_dataset,
                                             batch_size=32,
                                             num_workers=32,
                                             pin_memory=True,
                                             shuffle=True)
    
    kd = KnowledgeDist(teacher_model=teacher,
                       student_model=student,
                       parameters=student.classifier.parameters(),
                       dataloader=dataloader,

                       lr=1e-3)
    kd.train(device='cuda')
    kd.save_student("EffSiLUNoScaleClass.pt")
 

if __name__ == "__main__": 

    # train_original_to_pl_silu()
    train_original_to_slayer_cnn()
    #train_no_scale_classifier()