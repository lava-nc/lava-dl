# Lava DL

__`lava-dl`__ is a library of deep learning tools, which consists of `lava.lib.dl.slayer` and `lava.lib.dl.netx` for training and deployment of event-based deep neural networks on traditional as well as neuromorphic backends.

## Lava-dl Workflow

<p align="center">
<img src="https://user-images.githubusercontent.com/11490108/135362329-a6cf89e7-9d9e-42e5-9f33-102537463e63.png" alt="Drawing" style="max-height: 400px;"/>
</p>

## __`lava.lib.dl.slayer`__ 

`lava.lib.dl.slayer` is an enhanced version of [SLAYER](https://github.com/bamsumit/slayerPytorch). Most noteworthy enhancements are: support for _recurrent network structures_, a wider variety of _neuron models_ and _synaptic connections_ (a complete list of features is [here](https://github.com/lava-nc/lava-dl/blob/main/lib/dl/slayer/README.md)). This version of SLAYER is built on top of the [PyTorch](https://pytorch.org/) deep learning framework, similar to its predecessor. For smooth integration with Lava, `lava.lib.dl.slayer` supports exporting trained models using the platform independent __hdf5 network exchange__ format. 

In future versions, SLAYER will get completely integrated into Lava to train Lava Processes directly. This will eliminate the need for explicitly exporting and importing the trained networks. 

### Example Code

__Import modules__
```python
import lava.lib.dl.slayer as slayer
```
__Network Description__
```python
# like any standard pyTorch network
class Network(torch.nn.Module):
    def __init__(self):
        ...
        self.blocks = torch.nn.ModuleList([# sequential network blocks 
                slayer.block.sigma_delta.Input(sdnn_params), 
                slayer.block.sigma_delta.Conv(sdnn_params,  3, 24, 3),
                slayer.block.sigma_delta.Conv(sdnn_params, 24, 36, 3),
                slayer.block.rf_iz.Conv(rf_params, 36, 64, 3, delay=True),
                slayer.block.rf_iz.Conv(sdnn_cnn_params, 64, 64, 3, delay=True),
                slayer.block.rf_iz.Flatten(),
                slayer.block.alif.Dense(alif_params, 64*40, 100, delay=True),
                slayer.block.cuba.Recurrent(cuba_params, 100, 50),
                slayer.block.cuba.KWTA(cuba_params, 50, 50, num_winners=5)
            ])

    def forward(self, x):
        for block in self.blocks: 
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
        return x

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
```
__Training__
```python
net = Network()
...
for epoch in range(epochs):
    for i, (input, ground_truth) in enumerate(train_loader):
        out = net(input)
        ...
    for i, (input, ground_truth) in enumerate(test_loader):
        out = net(input)
        ...
```
__Export the network__
```python
net.export_hdf5('network.net')
```

## __`lava.lib.dl.netx`__ 

For inference using Lava, `lava.lib.dl.netx` provides an automated API for loading SLAYER-trained models as Lava Processes, which can be directly run on a desired backend. `lava.lib.dl.netx` imports models saved via SLAYER using the hdf5 network exchange format. The details of hdf5 network description specification can be found [here](https://github.com/lava-nc/lava-dl/blob/main/lib/dl/netx/README.md).

### Example Code

__Import modules__
```python
from lava.lib.dl.netx import hdf5
```
__Load the trained network__
```python
# Import the model as a Lava Process
net = hdf5.Network(net_config='network.net')
```
__Attach Processes for Input Injection and Output Readout__
```python
from lava.proc.io import InputLoader, BiasWriter, OutputReader

# Instantiate the processes
input_loader = InputLoader(dataset=testing_set)
bias_writer = BiasWriter(shape=input_shape)
output = OutputReader()

# Connect the input to the network:
input_loader.data_out.connect(bias_writer.bias_in)
bias_writer.bias_out.connect(net.in_layer.bias)

# Connect network-output to the output process
net.out_layer.neuron.s_out.connect(output.net_output_in)
```
__Run the network__
```python
from lava.magma import run_configs as rcfg
from lava.magma import run_conditions as rcnd

net.run(condition=rcnd.RunSteps(total_run_time), run_cfg=rcfg.Loihi1SimCfg())
```

