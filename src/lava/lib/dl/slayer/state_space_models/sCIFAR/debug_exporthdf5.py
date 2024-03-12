import torch
import torch.nn as nn
import torch.optim as optim

from lava.lib.dl.slayer.state_space_models.s4 import S4D
import os
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn.functional as F

import lava.lib.dl.slayer as slayer
import torch

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


batch_size = 1 
inp_dim = 8 
L = 5 # sequence length
dropout = 0
inp = torch.rand(batch_size, inp_dim, L) 

model = S4D(d_model=inp_dim,
            d_state=4,
            dropout=dropout,
            transposed=True,
            final_act = None,
            activation = None,
            lr=None,
            is_real=True)
model.__name__ = "S4D"
model.setup_step()


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        sdnn_params = { # sigma-delta neuron parameters
                'threshold'     : 0,    # delta unit threshold
                'tau_grad'      : 0.5,    # delta unit surrogate gradient relaxation parameter
                'scale_grad'    : 1,      # delta unit surrogate gradient scale parameter
                'requires_grad' : False,   # trainable threshold
                'shared_param'  : True,   # layer wise threshold
               # 'activation'    : model, # activation function
            ## 'activation' : F.relu

            }
        sdnn_S4d_params = { # conv layer has additional mean only batch norm
                **sdnn_params,                                 # copy all sdnn_params
                'activation' : model, # mean only quantized batch normalizaton
            }
        sdnn_dense_params = { # dense layers have additional dropout units enabled
                **sdnn_params,                        # copy all sdnn_cnn_params
                'activation' : F.relu
            }
        
        self.blocks = torch.nn.ModuleList([# sequential network blocks 
                slayer.block.sigma_delta.Input(sdnn_dense_params),
                slayer.block.sigma_delta.Dense(sdnn_S4d_params, in_neurons=inp_dim, out_neurons=inp_dim),
                slayer.block.sigma_delta.Output(sdnn_dense_params, in_neurons=inp_dim, out_neurons=inp_dim)
            ])
    

        #warum hat das keinen fehler geschmissen?
        self.blocks[0].synapse.weight.data = torch.eye(inp_dim).reshape((inp_dim,inp_dim,1,1,1))
        self.blocks[0].synapse.weight.requires_grad = False
        self.blocks[1].synapse.weight.data = torch.eye(inp_dim).reshape((inp_dim,inp_dim,1,1,1))
        self.blocks[1].synapse.weight.requires_grad = False
        #self.blocks[2].synapse.weight.data = torch.eye(inp_dim).reshape((inp_dim,inp_dim,1,1,1))
        #self.blocks[2].synapse.weight.requires_grad = False
        
    def forward(self, x):        
        for block in self.blocks: 
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
        return x

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

net = Network()
out_dl = net(inp)

from lava.lib.dl import netx
net.export_hdf5("debug1.net")
loaded_net =  netx.hdf5.Network(net_config='debug1.net')
