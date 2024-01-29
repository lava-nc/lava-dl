
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from lava.lib.dl.slayer.state_space_models.s4 import S4D
from lava.lib.dl.slayer.object_detection.models.yolo_kp import Network as YoloKP
from lava.lib.dl import slayer
import lava.lib.dl.slayer.image_classification.my_efficientnet as my_efficientnet
from lava.lib.dl.slayer.image_classification.efficientnet import MyEfficientNet, my_efficientnet_b0, ReLU1
import yaml
from lava.lib.dl.slayer.image_classification.piecewise_linear_silu import PiecewiseLinearSiLU

# class ReLU1(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return torch.clamp(x, min=0, max=1)

# def efficientnet_replace_activation(model, activation_fn):
#     for child_name, child in model.named_children():
#         if isinstance(child, nn.SiLU):
#             setattr(model, child_name, activation_fn())
#         elif  isinstance(child, nn.Sigmoid):
#             setattr(model, child_name, ReLU1())
#         elif len(list(child.children())) > 0:
#             # Recursively apply the function to child modules
#             efficientnet_replace_activation(child, activation_fn)

class EfficientNetLSTM(nn.Module):

    def __init__(self,
                 lstm_num_hidden=1280,
                 num_readout_hidden=64,
                 num_classes=60,
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.train()

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-2],
                                          nn.Conv2d(1280, 1280, kernel_size=(7,7)),
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
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.train()

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1],
                                          nn.Conv2d(1280, 1280, kernel_size=(7,7)),
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
                 readout_bias=True,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.train()

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1],
                                          nn.Flatten())

        # S4D Layer 
        self.s4d = S4D(d_model=s4d_num_hidden,
                       d_state=s4d_states,
                       dropout=0.0,
                       transposed=False,
                       lr=s4d_lr,
                       is_real=s4d_is_real)

        # Readout layer
        self.readout = nn.Sequential(nn.Linear(s4d_num_hidden, num_readout_hidden, bias=readout_bias),
                                     nn.ReLU(),
                                     nn.Linear(num_readout_hidden, num_classes, bias=readout_bias))


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
        x = self.s4d(x)

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
        x = self.s4d(x)

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
        x = self.s4d(x)

        # Take last step output from S4D and pass through readout layer
        x = x[:, -1, :]
        x = self.readout(x)

        return x



class YoloS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=64,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 yolo_model_path='network.pt',
                 yolo_args_path='args.txt',
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained YoloKP and remove last layer

        with open(yolo_args_path, "rt") as f:
            model_args = slayer.utils.dotdict(yaml.safe_load(f))

        print(model_args)

        self.yolo = YoloKP(threshold=model_args.threshold,
                           tau_grad=model_args.tau_grad,
                           scale_grad=model_args.scale_grad,
                           num_classes=11,
                           clamp_max=model_args.clamp_max).cuda()
        self.yolo.init_model((448, 448))
        self.yolo.load_state_dict(torch.load(yolo_model_path))

        self.yolo.input_blocks[0].neuron.delta.shape = None

        # Remove last layer
        #self.yolo = nn.Sequential(*list(self.yolo.children())[:-1])

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


        # Move batch into images dimension for yolo 
        x = x.movedim(1, -1)

        # Pass input through EfficientNet
        for block in self.yolo.input_blocks:
            x = block(x)
        for block in self.yolo.blocks:
            x = block(x)

        x = x.movedim(-1, 1).sum(-1).sum(-1)

        # Pass output through S4D layer
        x = self.s4d(x)

        # Take last step output from S4D and pass through readout layer
        x = x[:, -1, :]
        x = self.readout(x)

        return x

class EfficientNetSlayerS4D(nn.Module):

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

        # Load pre-trained EfficientNet and remove last layer
        # self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        # self.efficientnet.train()
        self.efficientnet = my_efficientnet.MyEfficientNet(my_efficientnet.inverted_residual_setting)
        # Use Efficientnet backbone but remove last layer
        # self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1],
        #                                   nn.Flatten())
        #self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-2],
        #                                  nn.Conv2d(1280, 1280, kernel_size=(7,7)),
        #                                  nn.Flatten())
        #rand_inp = torch.rand(3, 3, 224, 224, 1)
        #self.efficientnet(rand_inp)
        #checkpoint = torch.load("my_eff.pt")
        #self.efficientnet.load_state_dict(checkpoint)
        eff_torch = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.load_from_network(eff_torch)
        self.efficientnet.eval()

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


        # Move batch into images dimension for slayer
        x = x.movedim(1, -1)

        # Pass input through EfficientNet
        x = self.efficientnet.features(x) 

        x = x.movedim(-1, 1).sum(-1).sum(-1)

        # Pass output through S4D layer
        x = self.s4d(x)

        # Take last step output from S4D and pass through readout layer
        x = x[:, -1, :]
        x = self.readout(x)

        return x


class PlSiLUEfficientNetS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=64,
                 readout_bias=True,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = my_efficientnet_b0(weights='IMAGENET1K_V1', activation=PiecewiseLinearSiLU, scale_act=ReLU1)
        checkpoint = torch.load("EffSiLU.pt")
        self.efficientnet.load_state_dict(checkpoint)

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1],
                                          nn.Flatten())
        self.efficientnet.train()

        # S4D Layer 
        self.s4d = S4D(d_model=s4d_num_hidden,
                       d_state=s4d_states,
                       dropout=0.0,
                       transposed=False,
                       lr=s4d_lr,
                       is_real=s4d_is_real)

        # Readout layer
        self.readout = nn.Sequential(nn.Linear(s4d_num_hidden, num_readout_hidden, bias=readout_bias),
                                     nn.ReLU(),
                                     nn.Linear(num_readout_hidden, num_classes, bias=readout_bias))

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
        x = self.s4d(x)

        # Take last step output from S4D and pass through readout layer
        x = x[:, -1, :]
        x = self.readout(x)

        return x


class MiniEfficientNetS4D(nn.Module):

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

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.train()

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
        x = self.efficientnet.features[0](x)
        x = self.efficientnet.features[1](x)
        x = self.efficientnet.features[2](x)
        x = self.efficientnet.features[3](x)

        if len(inp_shape) == 5:
            x = x.reshape(inp_shape[0], inp_shape[1], *x.shape[1:]) # Get to dimension (B, T, C) 
        else:
            raise NotImplementedError("Not implement for unbatched data")
            x = x.reshape(inp_shape[0], -1)

        x = x.sum(-1).sum(-1)

        # Pass output through S4D layer
        x = self.s4d(x)

        # Take last step output from S4D and pass through readout layer
        x = x[:, -1, :]
        x = self.readout(x)

        return x

model_registry = {
    "efficientnet-b0-LSTM": EfficientNetLSTM,
    "efficientnet-b0-S4D": EfficientNetS4D,
    "efficientnet-b0-pl-silu-S4D": PlSiLUEfficientNetS4D,
    "mini-efficientnet-b0-S4D": MiniEfficientNetS4D,
    "efficientnet-b0-Slayer-S4D": EfficientNetSlayerS4D,
    "efficientnet-b0": EfficientNetAblation,
    "CNN-S4D": CNNS4D,
    "raw-S4D": BottleneckS4D,
    "YoloKP-S4D": YoloS4D, 
}
