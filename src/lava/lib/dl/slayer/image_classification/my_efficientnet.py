import torch
from torch import nn
from torch.nn import functional as F
from lava.lib.dl import slayer
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
import copy
from torch import Tensor

from torch.nn.utils.weight_norm import WeightNorm

class PiecewiseLinearSiLU(nn.Module):
    def __init__(self):
        super(PiecewiseLinearSiLU, self).__init__()

    def forward(self, x):
        x = x.clamp(-5., x.max().item())
        x[x<-1] = (x[x<-1] + 5) * -0.2
        return x


class Clamp6(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=-6, max=6)

class ReLU1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0, max=1)

class Ind(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x 

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        # if block is None:
            # block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))

bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
inverted_residual_setting = [
    bneck_conf(1, 3, 1, 32, 16, 1),
    bneck_conf(6, 3, 2, 16, 24, 2),
    bneck_conf(6, 5, 2, 24, 40, 2),
    bneck_conf(6, 3, 2, 40, 80, 3),
    bneck_conf(6, 5, 1, 80, 112, 3),
    bneck_conf(6, 5, 2, 112, 192, 4),
    bneck_conf(6, 3, 1, 192, 320, 1),
]
last_channel = None



class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        weight_scale: float = 1,
    ) -> None:
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
            'activation'    : Ind(),      # activation function
        }
        scale_sdnn_params= {
            **sigma_params,
            'activation'    : Clamp6(),      # activation function
        }
        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}
        scale_neuron_kwargs = {**scale_sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        self.fc1 = slayer.block.sigma_delta.Conv(neuron_kwargs, 
                                                 input_channels,
                                                 squeeze_channels,
                                                 kernel_size=1,
                                                 padding=0,
                                                 stride=1,
                                                 weight_scale=weight_scale,
                                                 **block_8_kwargs)
        self.fc2 = slayer.block.sigma_delta.Conv(scale_neuron_kwargs, 
                                                 squeeze_channels,
                                                 input_channels,
                                                 kernel_size=1,
                                                 padding=0,
                                                 stride=1,
                                                 weight_scale=weight_scale,
                                                 **block_8_kwargs)

    def forward(self, input: Tensor) -> Tensor:
        scale = self.fc1(input)
        scale = self.fc2(scale)
        return scale


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        # stochastic_depth_prob: float,
        # norm_layer: Callable[..., nn.Module],
        weight_scale,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        # self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        sigma_params = {  # sigma-delta neuron parameters
            'threshold'     : 0.0,   # delta unit threshold
            'tau_grad'      : 0.1,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : 0.1,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,       # trainable threshold
            'shared_param'  : True,        # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            # 'activation'    : nn.ReLU(),      # activation function
            'activation'    : PiecewiseLinearSiLU(),      # activation function
        }
        out_sdnn_params= {
            **sigma_params,
            'activation'    : Ind(),      # activation function
        }
        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}
        out_neuron_kwargs = {**out_sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}

        layers: List[nn.Module] = []

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                slayer.block.sigma_delta.Conv(neuron_kwargs, 
                                              cnf.input_channels,
                                              expanded_channels,
                                              kernel_size=1,
                                              padding=0,
                                              stride=1,
                                              weight_scale=weight_scale,
                                              **block_8_kwargs)
            )

        # depthwise
        layers.append(
            slayer.block.sigma_delta.Conv(neuron_kwargs, 
                                          expanded_channels,
                                          expanded_channels,
                                          kernel_size=cnf.kernel,
                                          padding=(cnf.kernel - 1) // 2,
                                          stride=cnf.stride,
                                          weight_scale=weight_scale,
                                          **block_8_kwargs)
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(SqueezeExcitation(expanded_channels,
                                        squeeze_channels, ))

        # project
        layers.append(
            slayer.block.sigma_delta.Conv(out_neuron_kwargs, 
                                          expanded_channels,
                                          cnf.out_channels,
                                          kernel_size=1,
                                          padding=0,
                                          stride=1,
                                          weight_scale=weight_scale,
                                          **block_8_kwargs)
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        return result




class MyEfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig],
        dropout: float = 0.2,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        sigma_params = {  # sigma-delta neuron parameters
            'threshold'     : 0.0,   # delta unit threshold
            'tau_grad'      : 0.1,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : 0.1,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,       # trainable threshold
            'shared_param'  : True,        # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            # 'activation'    : nn.ReLU(),      # activation function
            # 'activation'    : Ind(),      # activation function
            'activation'    : PiecewiseLinearSiLU(),      # activation function
        }
        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        block_8_kwargs = dict(weight_norm=True, delay_shift=False, pre_hook_fx=quantize_8bit)
        neuron_kwargs = {**sdnn_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}


        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            slayer.block.sigma_delta.Conv(neuron_kwargs,
                                          3,
                                          firstconv_output_channels, 
                                          kernel_size=3, 
                                          padding=1,
                                          stride=2,
                                          weight_scale=1,
                                          **block_8_kwargs),
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                # sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(MBConv(block_cnf, weight_scale=1))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            slayer.block.sigma_delta.Conv(neuron_kwargs,
                                          lastconv_input_channels,
                                          lastconv_output_channels, 
                                          kernel_size=1, 
                                          padding=0,
                                          stride=1,
                                          weight_scale=1,
                                          **block_8_kwargs),
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        self.detach_layers = True

    def _forward_impl(self, x: Tensor) -> Tensor:

        self.out_features = []
        for feat in self.features:
            if self.detach_layers:
                x = feat(x.detach())
            else:
                x = feat(x)
            self.out_features.append(x)

        x = self.avgpool(x.squeeze(-1))
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def load_from_network(self, net):

        for i in range(len(self.features)):
            my_feat = self.features[i]
            their_feat = net.features[i]
            if isinstance(my_feat, slayer.block.sigma_delta.Conv):
                MyEfficientNet.load_weights(my_feat, their_feat[0])
            elif isinstance(my_feat, nn.Sequential):
                MyEfficientNet.load_subset(my_feat, their_feat)
            else:
                raise Exception("Something went wrong while parsing the network")
        
        self.classifier[1].weight.data = net.classifier[1].weight.data
    
    def load_subset(children, their_children):
        for i, child in enumerate(children):
            their_child = their_children[i]
            if isinstance(child, slayer.block.sigma_delta.Conv):
                MyEfficientNet.load_weights(child, their_child[0])
            elif isinstance(child, MBConv):
                MyEfficientNet.load_subset(child.block, their_child.block)
            elif isinstance(child, SqueezeExcitation):
                MyEfficientNet.load_weights(child.fc1, their_child.fc1)
                MyEfficientNet.load_weights(child.fc2, their_child.fc2)
            else:
                raise Exception("Something went wrong while parsing the network")



    def load_weights(my_feat, their_feat):
        my_feat.synapse.disable_weight_norm()
        if their_feat.groups > 1:
            for g in range(their_feat.groups):
                my_feat.synapse.weight.data[:, g:g+1, :, :, 0] = their_feat.weight.data
                my_feat.synapse.weight.data[:, g:g+1, :, :, 0] = their_feat.weight.data
        else:
            my_feat.synapse.weight.data = their_feat.weight.data.unsqueeze(-1)
            my_feat.synapse.weight.data = their_feat.weight.data.unsqueeze(-1)

        my_feat.synapse.enable_weight_norm()



