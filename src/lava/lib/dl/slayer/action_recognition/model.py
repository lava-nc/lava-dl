
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from lava.lib.dl.slayer.state_space_models.s4 import S4D
from lava.lib.dl.slayer.object_detection.models.yolo_kp import Network as YoloKP
from lava.lib.dl import slayer
import lava.lib.dl.slayer.image_classification.my_efficientnet as my_efficientnet
from lava.lib.dl.slayer.image_classification.efficientnet import MyEfficientNet, my_efficientnet_b0
import yaml
from lava.lib.dl.slayer.image_classification.piecewise_linear_silu import PiecewiseLinearSiLU
from lava.lib.dl.slayer.image_classification.relu1 import ReLU1 


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
                 *args,
                 **kwargs):
        super().__init__()

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        self.efficientnet.train()

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


class PlSiLUNoScaleEfficientNetS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=64,
                 readout_bias=True,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 train_backbone=True,
                 *args,
                 **kwargs):
        super().__init__()

        self.train_backbone = train_backbone

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = my_efficientnet_b0(weights='IMAGENET1K_V1', activation=PiecewiseLinearSiLU, scale=False)
        checkpoint = torch.load("EffSiLUNoScale.pt")
        self.efficientnet.load_state_dict(checkpoint)

        if self.train_backbone:
            self.efficientnet.train()
        else:
            self.efficientnet.eval()

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
        if self.train_backbone:
            x = self.efficientnet(x)
        else:
            with torch.no_grad():
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

class LowResPlSiLUNoScaleEfficientNetS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=64,
                 readout_bias=True,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 train_backbone=True,
                 *args,
                 **kwargs):
        super().__init__()

        self.train_backbone = train_backbone

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = my_efficientnet_b0(weights='IMAGENET1K_V1',
                                               activation=PiecewiseLinearSiLU, 
                                               scale=False, 
                                               low_res=True)
        # checkpoint = torch.load("EffSiLUNoScale.pt")
        # self.efficientnet.load_state_dict(checkpoint)

        if self.train_backbone:
            self.efficientnet.train()
        else:
            self.efficientnet.eval()

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
        if self.train_backbone:
            x = self.efficientnet(x)
        else:
            with torch.no_grad():
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

class PlSiLUEfficientNetS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=64,
                 readout_bias=True,
                 num_classes=60,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 train_backbone=True,
                 *args,
                 **kwargs):
        super().__init__()

        self.train_backbone = train_backbone

        # Load pre-trained EfficientNet and remove last layer
        self.efficientnet = my_efficientnet_b0(weights='IMAGENET1K_V1', activation=PiecewiseLinearSiLU, scale_act=ReLU1)
        checkpoint = torch.load("EffSiLU.pt")
        self.efficientnet.load_state_dict(checkpoint)

        # Use Efficientnet backbone but remove last layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1],
                                          nn.Flatten())
        if self.train_backbone:
            self.efficientnet.train()
        else:
            self.efficientnet.eval()

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
        if self.train_backbone:
            x = self.efficientnet(x)
        else:
            with torch.no_grad():
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


class SlayerMBConv(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 squeeze_features: int,
                 out_features: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1):
        super().__init__()
        
        self.in_features = in_features

        sigma_params = {  # sigma-delta neuron parameters
            'threshold'     : 0.0,   # delta unit threshold
            'tau_grad'      : 0.1,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : 0.1,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,       # trainable threshold
            'shared_param'  : True,        # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : PiecewiseLinearSiLU(),      # activation function
        }
        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}
        if self.in_features:
            self.in_layer = slayer.block.sigma_delta.Conv(neuron_kwargs,
                                                             in_features=in_features,
                                                             out_features=hidden_features,
                                                             kernel_size=1, 
                                                             stride=1,
                                                             padding=0,
                                                             weight_scale=1,
                                                             **block_8_kwargs)

        self.se = []
        self.se.append(slayer.block.sigma_delta.Conv(neuron_kwargs,
                                                         in_features=hidden_features,
                                                         out_features=hidden_features,
                                                         kernel_size=kernel_size, 
                                                         stride=1,
                                                         padding=padding,
                                                         weight_scale=1,
                                                         groups=hidden_features,
                                                         **block_8_kwargs))

        self.se.append(slayer.block.sigma_delta.Conv(neuron_kwargs,
                                                         in_features=hidden_features,
                                                         out_features=squeeze_features,
                                                         kernel_size=1, 
                                                         stride=1,
                                                         padding=0,
                                                         weight_scale=1,
                                                         **block_8_kwargs))

        self.se.append(slayer.block.sigma_delta.Conv(neuron_kwargs,
                                                         in_features=squeeze_features,
                                                         out_features=hidden_features,
                                                         kernel_size=1, 
                                                         stride=1,
                                                         padding=0,
                                                         weight_scale=1,
                                                         **block_8_kwargs))
        self.se = nn.Sequential(*self.se)

        self.out_layer = slayer.block.sigma_delta.Conv(neuron_kwargs,
                                                         in_features=hidden_features,
                                                         out_features=out_features,
                                                         kernel_size=1, 
                                                         stride=stride,
                                                         padding=0,
                                                         weight_scale=1,
                                                         **block_8_kwargs)


    def forward(self, x):

            
        if self.in_features:
            x = self.in_layer(x)

        res = x

        x = self.se(x)

        x = x + res

        x = self.out_layer(x)

        return x 


class SlayerCNN(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 *args,
                 **kwargs):
        super().__init__()

        sigma_params = {  # sigma-delta neuron parameters
            'threshold'     : 0.0,   # delta unit threshold
            'tau_grad'      : 0.1,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : 0.1,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,       # trainable threshold
            'shared_param'  : True,        # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : PiecewiseLinearSiLU(),      # activation function
        }
        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        # Feature extraction
        layers: List[nn.Module] = []

        layers.append(
            slayer.block.sigma_delta.Conv(neuron_kwargs,
                                          in_features=3,
                                          out_features=32,
                                          kernel_size=5, 
                                          stride=2,
                                          padding=2,
                                          weight_scale=1,
                                          **block_8_kwargs),
        )

        layers.append(SlayerMBConv(in_features=None,
                                   hidden_features=32,
                                   squeeze_features=8, 
                                   out_features=32,
                                   stride=2,
                                   padding=1,
                                   kernel_size=3))

        layers.append(SlayerMBConv(in_features=None,
                                   hidden_features=32,
                                   squeeze_features=8, 
                                   out_features=128,
                                   stride=2,
                                   padding=1,
                                   kernel_size=3))

        layers.append(SlayerMBConv(in_features=None,
                                   hidden_features=128,
                                   squeeze_features=32, 
                                   out_features=320,
                                   stride=2,
                                   padding=1,
                                   kernel_size=3))

        layers.append(SlayerMBConv(in_features=None,
                                   hidden_features=320,
                                   squeeze_features=64, 
                                   out_features=1280,
                                   stride=2,
                                   padding=0,
                                   kernel_size=1))

                                   
        self.features = nn.Sequential(*layers)

        # Readout layer
        self.readout = nn.Linear(1280, num_classes)

    def forward(self, x):

        # Pass input through EfficientNet
        x = self.features(x.unsqueeze(-1)).squeeze(-1) 

        x = x.sum(-1).sum(-1)

        x = self.readout(x)

        return x

class SlayerCNNS4D(nn.Module):

    def __init__(self,
                 s4d_states=64,
                 s4d_num_hidden=1280,
                 num_readout_hidden=128,
                 num_classes=10,
                 s4d_is_real=True,
                 s4d_lr=1e-3,
                 readout_bias=True,
                 *args,
                 **kwargs):
        super().__init__()

        sigma_params = {  # sigma-delta neuron parameters
            'threshold'     : 0.0,   # delta unit threshold
            'tau_grad'      : 0.1,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : 0.1,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,       # trainable threshold
            'shared_param'  : True,        # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : PiecewiseLinearSiLU(),      # activation function
        }
        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        cnn = SlayerCNN(num_classes=1000)
        # TODO use best model
        # cnn.load_state_dict(torch.load("../image_classification/checkpoint.pth"))
        # Use cnn backbone but remove last layer
        self.features = nn.Sequential(*list(cnn.children())[:-1])


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

        # Move batch into images dimension for slayer
        x = x.movedim(1, -1)

        # Pass input through backbone 
        x = self.features(x) 

        x = x.movedim(-1, 1).sum(-1).sum(-1)

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
    "efficientnet-b0-pl-silu-noscale-S4D": PlSiLUNoScaleEfficientNetS4D,
    "efficientnet-b0-low-res-pl-silu-noscale-S4D": LowResPlSiLUNoScaleEfficientNetS4D,
    "mini-efficientnet-b0-S4D": MiniEfficientNetS4D,
    "efficientnet-b0-Slayer-S4D": EfficientNetSlayerS4D,
    "efficientnet-b0": EfficientNetAblation,
    "CNN-S4D": CNNS4D,
    "raw-S4D": BottleneckS4D,
    "YoloKP-S4D": YoloS4D, 
    "SlayerCNN": SlayerCNN,
    "SlayerCNNS4D": SlayerCNNS4D,
}
