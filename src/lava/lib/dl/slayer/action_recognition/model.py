
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from lava.lib.dl.slayer.state_space_models.s4 import S4D

class ReLU1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0, max=1)

def efficientnet_replace_activation(model, activation_fn):
    for child_name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            setattr(model, child_name, activation_fn())
        elif  isinstance(child, nn.Sigmoid):
            setattr(model, child_name, ReLU1())
        elif len(list(child.children())) > 0:
            # Recursively apply the function to child modules
            efficientnet_replace_activation(child, activation_fn)

class EfficientNetLSTM(nn.Module):

    def __init__(self,
                 lstm_num_hidden=1280,
                 num_readout_hidden=64,
                 num_classes=60,
                 efficientnet_activation='silu',
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.train()

        if efficientnet_activation == "relu":
            efficientnet_replace_activation(self.efficientnet,
                                            nn.ReLU)
        elif efficientnet_activation == "silu":
            pass
        else:
            raise NotImplementedError("Only relu und silu are implemented.")

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1],
                                          nn.Flatten())

        # LSTM layer
        self.lstm = nn.LSTM(input_size=1280, hidden_size=lstm_num_hidden)

        # Readout layer
        self.readout = nn.Sequential(nn.Linear(lstm_num_hidden, num_readout_hidden),
                                     nn.ReLU(),
                                     nn.Linear(num_readout_hidden, num_classes))


    def forward(self, x):


        inp_shape = x.shape

        # Move batch into images dimension for efficientnet
        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0] * inp_shape[1], *inp_shape[2:])
        else:
            x = x.squeeze(0)

        # Pass input through EfficientNet
        x = self.efficientnet(x)

        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0], inp_shape[1], *x.shape[1:]).transpose(0, 1)
        else:
            x = x.unsqueeze(0)

        # Pass output through LSTM layer
        x = self.lstm(x)[0]

        # Take last step output from LSTM and pass through readout layer
        x = x[-1, :, :]
        x = self.readout(x)

        return x

class EfficientNetAblation(nn.Module):

    def __init__(self,
                 num_readout_hidden=64,
                 num_classes=60,
                 efficientnet_activation='silu',
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.train()

        if efficientnet_activation == "relu":
            efficientnet_replace_activation(self.efficientnet,
                                            nn.ReLU)
        elif efficientnet_activation == "silu":
            pass
        else:
            raise NotImplementedError("Only relu und silu are implemented.")

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1],
                                          nn.Flatten())


        # Readout layer
        self.readout = nn.Sequential(nn.Linear(1280, num_readout_hidden),
                                     nn.ReLU(),
                                     nn.Linear(num_readout_hidden, num_classes))


    def forward(self, x):


        inp_shape = x.shape

        # Move batch into images dimension for efficientnet
        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0] * inp_shape[1], *inp_shape[2:])
        else:
            x = x.squeeze(0)

        # Pass input through EfficientNet
        x = self.efficientnet(x)

        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0], inp_shape[1], *x.shape[1:]).transpose(0, 1)
        else:
            x = x.unsqueeze(0)

        # Take last step output from Efficientnet and pass through readout layer
        x = x[-1, :, :]
        x = self.readout(x)

        return x



class EfficientNetS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=64,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 efficientnet_activation='silu',
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.train()

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1],
                                          nn.Flatten())

        if efficientnet_activation == "relu":
            efficientnet_replace_activation(self.efficientnet,
                                            nn.ReLU)
        elif efficientnet_activation == "silu":
            pass
        else:
            raise NotImplementedError("Only relu und silu are implemented.")

        # S4D Layer 
        self.s4d = S4D(d_model=s4d_num_hidden,
                       d_state=s4d_states,
                       dropout=0.0,
                       transposed=False,
                       lr=s4d_lr,
                       is_real=s4d_is_real)

        # Readout layer
        self.readout = nn.Sequential(nn.Linear(s4d_num_hidden, num_readout_hidden),
                                     nn.ReLU(),
                                     nn.Linear(num_readout_hidden, num_classes))


    def forward(self, x):


        inp_shape = x.shape

        # Move batch into images dimension for efficientnet
        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0] * inp_shape[1], *inp_shape[2:])
        else:
            x = x.squeeze(0)

        # Pass input through EfficientNet
        x = self.efficientnet(x)

        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0], inp_shape[1], *x.shape[1:]) # Get to dimension (B, T, C) 
        else:
            raise NotImplementedError("Not implement for unbatched data")
            x = x.reshape(inp_shape[0], -1)

        # Pass output through S4D layer
        x = self.s4d(x)[0]

        # Take last step output from S4D and pass through readout layer
        x = x[:, -1, :]
        x = self.readout(x)

        return x

class CNNS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=64,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 *args,
                 **kwargs):
        super().__init__()

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), 
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(),
                                          nn.Dropout2d(p=0.2),
                                          nn.Conv2d(32, 64, 3, 2, 1, bias=False), 
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(),
                                          nn.Dropout2d(p=0.2),
                                          nn.Conv2d(64, 128, 3, 2, 1, bias=False), 
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(),
                                          nn.Dropout2d(p=0.2),
                                          nn.Conv2d(128, 512, 3, 2, 1, bias=False), 
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(),
                                          nn.Dropout2d(p=0.2),
                                          nn.Conv2d(512, 1280, 3, 2, 1, bias=False), 
                                          nn.BatchNorm2d(1280),
                                          nn.ReLU(),
                                          nn.Dropout2d(p=0.2),
                                          nn.AdaptiveAvgPool2d(output_size=1), 
                                          nn.Flatten())

        # S4D layer
        self.s4d = S4D(d_model=s4d_num_hidden,
                       d_state=s4d_states,
                       dropout=0.0,
                       transposed=False,
                       lr=s4d_lr,
                       is_real=s4d_is_real)

        # Readout layer
        self.readout = nn.Sequential(nn.Linear(s4d_num_hidden, num_readout_hidden),
                                     nn.ReLU(),
                                     nn.Linear(num_readout_hidden, num_classes))


    def forward(self, x):


        inp_shape = x.shape

        # Move batch into images dimension for efficientnet
        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0] * inp_shape[1], *inp_shape[2:])
        else:
            x = x.squeeze(0)

        # Pass input through EfficientNet
        x = self.efficientnet(x)

        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0], inp_shape[1], *x.shape[1:]) # Get to dimension (B, T, C) 
        else:
            raise NotImplementedError("Not implement for unbatched data")
            x = x.reshape(inp_shape[0], -1)

        # Pass output through S4D layer
        x = self.s4d(x)[0]

        # Take last step output from S4D and pass through readout layer
        x = x[:, -1, :]
        x = self.readout(x)

        return x


class BottleneckS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=64,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 *args,
                 **kwargs):
        super().__init__()

        self.bottleneck = nn.Linear(224*224*3, s4d_num_hidden)

        # S4D layer
        self.s4d = S4D(d_model=s4d_num_hidden,
                       d_state=s4d_states,
                       dropout=0.0,
                       transposed=False,
                       lr=s4d_lr,
                       is_real=s4d_is_real)

        # Readout layer
        self.readout = nn.Sequential(nn.Linear(s4d_num_hidden, num_readout_hidden),
                                     nn.ReLU(),
                                     nn.Linear(num_readout_hidden, num_classes))


    def forward(self, x):


        inp_shape = x.shape

        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0], inp_shape[1], -1) # Get to dimension (B, T, C) 
        else:
            raise NotImplementedError("Not implement for unbatched data")
            x = x.reshape(inp_shape[0], -1)
        
        x = self.bottleneck(x)

        # Pass output through S4D layer
        x = self.s4d(x)[0]

        # Take last step output from S4D and pass through readout layer
        x = x[:, -1, :]
        x = self.readout(x)

        return x



model_registry = {
    "efficientnet-b0-LSTM": EfficientNetLSTM,
    "efficientnet-b0-S4D": EfficientNetS4D,
    "efficientnet-b0": EfficientNetAblation,
    "CNN-S4D": CNNS4D,
    "raw-S4D": BottleneckS4D,
}
