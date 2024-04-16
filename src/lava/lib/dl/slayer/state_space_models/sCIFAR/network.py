import warnings
from lava.lib.dl.slayer.state_space_models.s4 import S4D
import torch
import lava.lib.dl.slayer as slayer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import h5py

class SCIFARNetwork(torch.nn.Module):
    # Loihi compatible network for sCIFAR classification
    # To-Do before Training
        # Classification is based on mean of last S4D + Activiation layer!
    def __init__(self, d_model=128, n_states=64, dropout=0, s4d_learning_rate=0.01, relu_in_s4d=False, skip=False, num_layers: int = 1):
        super(SCIFARNetwork, self).__init__()
        if skip: 
            raise(NotImplementedError, "Skip connection currently not enabled.")
        
        self.d_model = d_model
        self.n_states = n_states
        self.dropout = dropout
        self.s4d_learning_rate = s4d_learning_rate

        if relu_in_s4d:
            self.activation = "relu"
            warnings.warn("Relu activation inside S4D is currently not implemented on Loihi/integrated in netx.")
        else:
            self.activation = None

        # We need four different instances of the S4D model 
        self.s4dmodels = [S4D(self.d_model, 
                            activation=None,
                            dropout=self.dropout,
                            transposed=True,
                            d_state=self.n_states,
                            final_act=None,
                            is_real = True,
                            skip=False,
                            lr=min(0.001, self.s4d_learning_rate)) for _ in range(num_layers)] 
        for model in self.s4dmodels:
            model.__name__ = "S4D"
            model.setup_step()
       
        identity = nn.Identity()
        identity.__name__ = "identity"

        def quantize_8bit(x: torch.tensor,
                          scale: int = (1 << 6),
                          descale: bool = False) -> torch.tensor:
            return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                 num_bits=8, descale=descale)

        sdnn_params = { # sigma-delta neuron parameters
                'threshold'     : 0,    # delta unit   #1/64
                'tau_grad'      : 0.5,    # delta unit surrogate gradient relaxation parameter
                'scale_grad'    : 1,      # delta unit surrogate gradient scale parameter
                'requires_grad' : False,   # trainable threshold
                'shared_param'  : True,   # layer wise threshold
                'dropout' : slayer.neuron.Dropout(p=self.dropout), # neuron dropout
                'norm' : slayer.neuron.norm.MeanOnlyBatchNorm, # mean only quantized batch normalizaton
            }


        s4d_params = [{**sdnn_params, "activation" : model} for model in self.s4dmodels]
        standard_params ={**sdnn_params, "activation" : identity}
        final_act_params = {**sdnn_params, "activation" : F.relu}
        
        self.blocks = [slayer.block.sigma_delta.Input(standard_params),
                       slayer.block.sigma_delta.Dense(standard_params, 3, self.d_model), # Expand model dim
                      ]


        for i in range(num_layers):
            s4d = slayer.block.sigma_delta.Dense(s4d_params[i], self.d_model, self.d_model, pre_hook_fx=quantize_8bit)
            s4d_reduction = slayer.block.sigma_delta.Dense(final_act_params, self.d_model, self.d_model, pre_hook_fx=quantize_8bit)
            ff = slayer.block.sigma_delta.Dense(final_act_params, self.d_model, self.d_model, weight_scale=3, weight_norm=True, pre_hook_fx=quantize_8bit)
            
            s4d.synapse.weight.data = torch.eye(self.d_model).reshape((self.d_model, self.d_model,1,1,1))
            s4d.synapse.weight.requires_grad = False

            s4d_reduction.synapse.weight.data = torch.eye(self.d_model).reshape((self.d_model, self.d_model,1,1,1))
            s4d_reduction.synapse.weight.requires_grad = False

            # ff.synapse.weight.data = torch.eye(self.d_model).reshape((self.d_model, self.d_model,1,1,1))

            self.blocks.append(s4d)
            self.blocks.append(s4d_reduction)
            self.blocks.append(ff)
            
        self.blocks.append(slayer.block.sigma_delta.Output(standard_params, self.d_model, 10))
        self.blocks = torch.nn.ModuleList(self.blocks)
                 
    def forward(self, x):
        x = x.transpose(-1, -2)
        for _, block in enumerate(self.blocks): 
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
        return x.mean(-1)
        
    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()
        return grad
        
    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))