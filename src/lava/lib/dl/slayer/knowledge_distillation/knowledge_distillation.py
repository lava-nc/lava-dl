import torch
from torch import nn, Tensor
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import Adam
from typing import Iterable, Callable, Dict, List
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

    

class FeatureExtractor(nn.Module):
    """
    Wrapper for nn.Module applying hooks for each layer.
    The hooks collect the output of each layer to be used in KD.

    """
    def __init__(self, model: nn.Module, layers: List[nn.Module]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {i: torch.empty(0) for i, layer in enumerate(layers)}
        self._hooks = []

        # for i, layer in enumerate(layers):
        #     self._hooks.append(layer.register_forward_hook(self.save_outputs_hook(i)))
    
    def register_hook(self, layer_id: int):
        self._hooks.append(self.layers[layer_id].register_forward_hook(self.save_outputs_hook(layer_id)))

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

    train_dataloader: torch.utils.data.Dataloader
    Dataloader to be used for input to both, student and teacher while training.

    val_dataloader: torch.utils.data.Dataloader
    Dataloader to be used for input to both, student and teacher for evaluation.

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
                 train_dataloader,
                 val_dataloader,
                 lr=1e-3,
                 n_epochs=1) -> None:
        self.teacher = FeatureExtractor(teacher_model, teacher_features)
        self.student = FeatureExtractor(student_model, student_features)
        self.num_features = len(teacher_features)
        self.lr = lr
        self.opt = Adam
        self.crit = MSELoss()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.n_epochs = n_epochs
    
    def prepare_feature(self, feat_idx):
        self.teacher.remove_hooks()
        self.student.remove_hooks()
        self.teacher.register_hook(feat_idx)
        self.student.register_hook(feat_idx)
    
    def train(self, device='cuda'):
        for feat_idx in range(self.num_features):
            self.prepare_feature(feat_idx)
            print(f"train {feat_idx=}")
            opt = self.opt(self.student.parameters(), lr=self.lr)
            for epoch in range(self.n_epochs):
                self.student.train()
                print(f"start {epoch=}")
                epoch_loss = 0
                for batch_idx, batch in enumerate(self.train_dataloader):
                    inputs, targets = batch 
                    inputs = inputs.to(device)

                    with torch.no_grad():
                        teacher_out = self.teacher(inputs)[feat_idx].detach()
                    opt.zero_grad()
                    student_out = self.student(inputs)[feat_idx]

                    if not teacher_out.shape == student_out.shape:
                        # TODO this can't stay here
                        student_out = student_out.movedim(-1, 1)
                        student_out = student_out.reshape(teacher_out.shape)

                    loss = self.crit(teacher_out, student_out)
                    epoch_loss = (epoch_loss * batch_idx + loss.item()) / (batch_idx+1)

                    loss.backward()
                    opt.step()
                    if batch_idx % 10 == 0:
                        print(f"{epoch=}, {batch_idx=}, {feat_idx=}, {loss.item()=}, {epoch_loss=}")
            
                with torch.no_grad():
                    self.student.eval()
                    val_epoch_loss = 0
                    for batch_idx, batch in enumerate(self.val_dataloader):
                        inputs, targets = batch 
                        inputs = inputs.to(device)
                        with torch.no_grad():
                            student_out = self.student(inputs)[feat_idx]
                            teacher_out = self.teacher(inputs)[feat_idx]

                            if not teacher_out.shape == student_out.shape:
                                # TODO this can't stay here
                                student_out = student_out.movedim(-1, 1)
                                student_out = student_out.reshape(teacher_out.shape)
                            loss = self.crit(teacher_out, student_out)
                            val_epoch_loss = (val_epoch_loss * batch_idx + loss.item()) / (batch_idx+1)
                            if batch_idx % 10 == 0:
                                print(f"val {epoch=}, {batch_idx=}, {feat_idx=}, {loss.item()=}, {val_epoch_loss=}")

    def save_student(self, fn):
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

    train_dataloader: torch.utils.data.Dataloader
    Dataloader to be used for input to both, student and teacher while training. 

    val_dataloader: torch.utils.data.Dataloader
    Dataloader to be used for input to both, student and teacher for evaluation.

    lr: float
    Learning rate. Default 1e-3

    n_epochs: int
    Number of epochs to train. Default 1.
    """
    def __init__(self, 
                 teacher_model,
                 student_model,
                 parameters,
                 train_dataloader,
                 val_dataloader,
                 lr=1e-3,
                 n_epochs=1,
                 print_interval=100) -> None:
        self.teacher = teacher_model
        self.teacher.eval()
        self.student = student_model
        self.student.train()
        self.lr = lr
        self.opt = Adam(parameters, lr=self.lr)
        self.crit = CrossEntropyLoss()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.n_epochs = n_epochs 
        self.print_interval = print_interval
        self.best_acc = 0

    def train(self, device='cuda'):
        for epoch in range(self.n_epochs):
            preds = []
            tgts = []
            self.student.train()
            for batch_idx, batch in enumerate(self.train_dataloader):
                inputs, targets = batch 
                inputs = inputs.to(device)

                with torch.no_grad():
                    teacher_out = self.teacher(inputs)
                self.opt.zero_grad()
                student_out = self.student(inputs)

                loss = self.crit(student_out, teacher_out.softmax(1))

                pred = torch.argmax(student_out, dim=1).detach().cpu().numpy().tolist()

                preds += pred
                tgts += targets.cpu().numpy().tolist()

                if batch_idx % self.print_interval == 0:
                    print(targets.cpu().numpy(), teacher_out.argmax(1).cpu().numpy(), student_out.argmax(1).cpu().numpy())
                    print(epoch, batch_idx, loss.item(), accuracy_score(tgts, preds))
                
                loss.backward()
                self.opt.step()

            val_preds = []
            val_tgts = []
            self.student.eval()
            for batch_idx, batch in enumerate(self.val_dataloader):
                inputs, targets = batch 
                inputs = inputs.to(device)
                with torch.no_grad():
                    student_out = self.student(inputs)
                    pred = torch.argmax(student_out, dim=1).detach().cpu().numpy().tolist()
                    val_preds += pred
                    val_tgts += targets.cpu().numpy().tolist()

                if batch_idx % self.print_interval == 0:
                    print("val", targets, student_out.argmax(1).cpu().numpy())
                    print(epoch, batch_idx, accuracy_score(val_tgts, val_preds))
            
            acc = accuracy_score(val_tgts, val_preds)
            if acc > self.best_acc:
                self.best_acc = acc
                self.save_student("checkpoint.pth")

            print("val", targets, student_out.argmax(1).cpu().numpy())
            print(epoch, batch_idx, acc, self.best_acc)


    def save_student(self, fn):
        self.student.eval()
        torch.save(self.student.state_dict(), fn) 

