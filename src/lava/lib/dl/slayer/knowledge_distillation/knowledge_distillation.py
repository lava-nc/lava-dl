import torch
from torch import nn, Tensor
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from typing import Iterable, Callable, Dict, List


    

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
    """
    Trains each layer of a student network on the output of a teacher network using MSE loss and Adam optimizer.

    The KD class creates hooks on each layer to obtain their output. Then it applies MSE loss between the output of the student and teacher.
    The teacher stays static and is not trained while the student gets trained layer-by-layer.

    Number of layers and their shape in the student model must be equal to the teacher.

    Arguments
    ---------
    teacher_model: torch.Module
    Teacher model

    student_model: torch.Module
    Student model

    teacher_features: List[toch.Module]
    List of layers to be used for training the student

    student_features: List[toch.Module]
    List of layers to be used for training.

    dataloader: torch.utils.data.Dataloader
    Dataloader to be used for input to both, student and teacher. No labels required.

    lr: float
    Learning rate. Default 1e-3

    n_epochs: int
    Number of epochs to train. Default 1.
    """

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
            opt = self.opt(self.student.parameters(), lr=self.lr)
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
                        print(f"{epoch=}, {batch_idx=}, {feat_idx=}, {loss.item()=}")

    def save_student(self, fn):
        self.student.remove_hooks()
        self.student.model.eval()
        torch.save(self.student.model.state_dict(), fn) 

class KnowledgeDist:
    """
    Trains a student network on the output/logits of a teacher network using Cross Entropy loss and Adam optimizer.

    The KD applies CE loss between the output of the student and the teacher network.
    The teacher stays static and is not trained while the student gets trained.

    The shape of the student output must matche the teacher.

    Arguments
    ---------
    teacher_model: torch.Module
    Teacher model

    student_model: torch.Module
    Student model

    teacher_features: List[toch.Module]
    List of layers to be used for training the student

    student_features: List[toch.Module]
    List of layers to be used for training.

    dataloader: torch.utils.data.Dataloader
    Dataloader to be used for input to both, student and teacher. No labels required.

    lr: float
    Learning rate. Default 1e-3

    n_epochs: int
    Number of epochs to train. Default 1.
    """
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
        for epoch in range(self.n_epochs):
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

