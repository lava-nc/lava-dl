from torchvision.models import efficientnet_b0, EfficientNet
from lava.lib.dl.slayer.image_classification.efficientnet import my_efficientnet_b0, MyEfficientNet
import importlib
import torch
from torch import nn, Tensor
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from lava.lib.dl import slayer
from torchvision.datasets import ImageNet
from sklearn.metrics import accuracy_score
from torchvision import transforms
from PIL import Image
from piecewise_linear_silu import PiecewiseLinearSiLU
from typing import Iterable, Callable, Dict, List


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
        return self.preprocess(self._load_image(self.fns[idx]))
    
    def __len__(self):
        return len(self.fns)
    

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: List[nn.Module]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {i: torch.empty(0) for i, layer in enumerate(layers)}
        self._hooks = []

        for i, layer in enumerate(layers):
            self._hooks.append(layer.register_forward_hook(self.save_outputs_hook(i)))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn
    
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features

class FeatureWiseKnowledgeDist:

    def __init__(self, 
                 teacher_model,
                 student_model,
                 teacher_features,
                 student_features,
                 dataloader,
                 lr=1e-3,
                 n_epochs=1) -> None:
        self.teacher = FeatureExtractor(teacher_model, teacher_features)
        self.student = FeatureExtractor(student_model, student_features)
        self.num_features = len(teacher_features)
        self.lr = lr
        self.opt = Adam
        self.crit = MSELoss()
        self.dataloader = dataloader
        self.n_epochs = n_epochs 
    
    def train(self, device='cuda'):
        self.student.train()
        for feat_idx in range(self.num_features):
            opt = Adam(self.student.parameters(), lr=self.lr)
            for epoch in range(self.n_epochs):
                for batch_idx, inputs in enumerate(self.dataloader):
                    inputs = inputs.to(device)

                    with torch.no_grad():
                        teacher_out = self.teacher(inputs)[feat_idx].detach()
                    opt.zero_grad()
                    student_out = self.student(inputs)[feat_idx]

                    loss = self.crit(teacher_out, student_out)

                    loss.backward()
                    opt.step()
                    if batch_idx % 100 == 0:
                        print(epoch, batch_idx, feat_idx, loss.item())

    def save_student(self, fn):
        self.student.remove_hooks()
        self.student.model.eval()
        torch.save(self.student.model.state_dict(), fn) 

class KnowledgeDist:
    def __init__(self, 
                 teacher_model,
                 student_model,
                 parameters,
                 dataloader,
                 lr=1e-3,
                 n_epochs=1) -> None:
        self.teacher = teacher_model
        self.teacher.eval()
        self.student = student_model
        self.student.train()
        self.lr = lr
        self.opt = Adam(parameters, lr=self.lr)
        self.crit = CrossEntropyLoss()
        self.dataloader = dataloader
        self.n_epochs = n_epochs 

    def train(self, device='cuda'):
        for epoch in range(1):
            for batch_idx, inputs in enumerate(self.dataloader):
                inputs = inputs.to(device)

                with torch.no_grad():
                    teacher_out = self.teacher(inputs)
                self.opt.zero_grad()
                student_out = self.student(inputs)

                loss = self.crit(student_out, teacher_out.softmax(1))
                if batch_idx % 100 == 0:
                    print(teacher_out.argmax(1).cpu().numpy(), student_out.argmax(1).cpu().numpy())
                    print(epoch, batch_idx, loss.item())
                
                loss.backward()
                self.opt.step()


    def save_student(self, fn):
        self.student.eval()
        torch.save(self.student.state_dict(), fn) 


class ReLU1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0, max=1)


def train_original_to_pl_silu():

    teacher = efficientnet_b0(weights='IMAGENET1K_V1').cuda()
    teacher.eval()
    
    student = my_efficientnet_b0(weights='IMAGENET1K_V1',
                                 activation=PiecewiseLinearSiLU,
                                 scale_act=ReLU1).cuda()
    
    imagenet_dataset = ImageNetDataset(file_list="/nas-data/pweidel/datasets/imagenet/index_file_train.txt", split='train')
    dataloader = torch.utils.data.DataLoader(dataset=imagenet_dataset,
                                             batch_size=32,
                                             num_workers=32,
                                             pin_memory=True,
                                             shuffle=True)
    
    fwkd = FeatureWiseKnowledgeDist(teacher, student, teacher.features, student.features, dataloader, lr=1e-3)
    fwkd.train()
    fwkd.save_student("EffSiLU.pt")
    

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
    train_no_scale_classifier()