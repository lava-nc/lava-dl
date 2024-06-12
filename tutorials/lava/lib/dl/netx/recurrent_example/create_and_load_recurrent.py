import torch
from lava.lib.dl import netx
from lava.lib.dl import slayer
import h5py
import os
from lava.proc import io
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
import numpy as np

class Network(torch.nn.Module):
    
    def __init__(self,):
        super(Network, self).__init__()
    
        self.cuba_params = {
                    'threshold'    : 1.0,
                    'current_decay': 0.5,
                    'voltage_decay': 0.15,
                    'tau_grad'     : 1., 
                    'scale_grad'   : 1., 
                    'shared_param' : False, 
                    'requires_grad': False, 
                    'graded_spike' : False,
                }
        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Recurrent(self.cuba_params, 6, 5, weight_scale=1.0, pre_hook_fx=None),
                slayer.block.cuba.Recurrent(self.cuba_params, 5, 5, weight_scale=1.0, pre_hook_fx=None),
            ])

        self.blocks[0].input_synapse.weight.data *= 0.
        self.blocks[0].input_synapse.weight.data += 0.1
        self.blocks[0].recurrent_synapse.weight.data *= 0.
        self.blocks[0].recurrent_synapse.weight.data += 0.05
        
        self.blocks[1].input_synapse.weight.data *= 0.
        self.blocks[1].input_synapse.weight.data += 0.1
        self.blocks[1].recurrent_synapse.weight.data *= 0.
        self.blocks[1].recurrent_synapse.weight.data += 0.05
        
    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike
    
    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
        
if __name__ == '__main__':
    
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    num_steps = 101
    input = torch.zeros((1,6,num_steps)) # batch = 1, channels = 6, time = 101
    input += 0.2
    
    net = Network()
    lava_dl_output = net(input).detach().numpy()
    
    filename = os.path.join(current_file_directory, 'recurrent_example.net')
    net.export_hdf5(filename)
    
    net_lava = netx.hdf5.Network(filename, input_message_bits=24)
    
    net_lava_input_scale_factor_exp = 10
    net_lava_input_scale_factor = 2**net_lava_input_scale_factor_exp
    net_lava.layers[0].synapse_rec.proc_params._parameters['weight_exp'] += net_lava_input_scale_factor_exp
    net_lava.layers[0].neuron.vth.init *= net_lava_input_scale_factor
    net_lava.layers[0].neuron.bias_mant.init *= net_lava_input_scale_factor

    input_lava = input[0].numpy() * net_lava_input_scale_factor
    
    source = io.source.RingBuffer(data=input_lava)
    sink = io.sink.RingBuffer(shape=net_lava.out.shape, buffer=num_steps+1)
    source.s_out.connect(net_lava.inp)
    net_lava.out.connect(sink.a_in)
    
    run_config = Loihi2SimCfg(select_tag='fixed_pt')
    run_condition = RunSteps(num_steps=num_steps)
    net_lava.run(condition=run_condition, run_cfg=run_config)
    lava_output = sink.data.get()
    net_lava.stop()
    
    print(f'lava-dl output spikes \n{lava_dl_output[0][0]=}')
    print(f'lava output spikes \n{lava_output[0]=}')
    
    print(f'lava-dl mean firing rate {lava_dl_output[0][0].mean()=}')
    print(f'lava mean firing rate {lava_output[0].mean()=}')
    
    eps = 0.02
    if np.abs(lava_dl_output[0][0].mean() - lava_output[0].mean()) >= eps:
        assert False, 'Mean firing rate mismatch'
    
    print('done')
    
