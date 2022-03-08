# Lava DL

__`lava-dl`__ is a library of deep learning tools within Lava that support  offline training, online training and inference methods for various Deep Event-Based Networks.

There are two main strategies for training Deep Event-Based Networks: _direct training_ and _ANN to SNN converison_. 

Directly training the network utilizes the information of precise timing of events. Direct training is very accurate and results in efficient networks. However, directly training networks take a lot of time and resources.

On the other hand, ANN to SNN conversion is especially suitable for rate coded SNNs where we can leverage fast training of ANNs. These converted SNNs, however, typically require increased latency compared to directly trained SNNs.

Lava-DL provides an improved version of [SLAYER](https://github.com/bamsumit/slayerPytorch) for direct training of deep event based networks and a new ANN-SNN accelerated training approach called [Bootstrap](https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/bootstrap/README.md) to mitigate high latency issue of conventional ANN-SNN methods for training Deep Event-Based Networks.

The lava-dl training libraries are independent of the core lava library since Lava Processes cannot be trained directly at this point. Instead, lava-dl is first used to train the model which can then be converted to a network of Lava processes using the netx library using platform independent hdf5 network description.

The library presently consists of

1. `lava.lib.dl.slayer` for natively training Deep Event-Based Networks.
2. `lava.lib.dl.bootstrap` for training rate coded SNNs.
3. `lava.lib.dl.netx` for training and deployment of event-based deep neural networks on traditional as well as neuromorphic backends.

More tools will be added in the future.

## Lava-DL Workflow

<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/140595634-a97886c6-280a-4771-830b-ae47a9324612.png" alt="Drawing" style="max-height: 400px;"/>
</p>

Typical Lava-DL workflow:
* **Training:** using `lava.lib.dl.{slayer/bootstrap}` which results in a _hdf5 network description_. Training usually follows an iterative cycle of architecture design, hyperparameter tuning, and backpropagation training.
* **Inference:** using `lava.lib.dl.netx` which generates lava proces from the hdf5 network description of the trained network and enables inference on different backends.

## Installation

### Cloning Lava-DL and Running from Source

We highly recommend cloning the repository and using pybuilder to setup lava.
 You will need to install pybuilder for the same.

**Note:** We assume you have already setup Lava with virtual environment. Make sure `PYTHONPATH` contains path to Lava core library first.

* Linux/MacOS: `echo $PYTHONPATH`
* Windows: `echo %PYTHONPATH%`

The output should contain something like this `/home/user/lava`

#### [Linux/MacOS]
```bash
$ git clone git@github.com:lava-dl/lava.git
$ cd lava-dl
$ pip install -r build-requirements.txt
$ pip install -r requirements.txt
$ export PYTHONPATH=$PYTHONPATH:$(pwd)/src
$ pyb -E unit
```
#### [Windows]
```cmd
cd %HOMEPATH%
git clone git@github.com:lava-dl/lava.git
cd lava-dl
pip install -r build-requirements.txt
pip install -r requirements.txt
set PYTHONPATH=%PYTHONPATH%;%cd%\src
pyb -E unit
```

You should expect the following output after running the unit tests:
```
PyBuilder version 0.13.3
Build started at 2021-11-05 18:44:51
------------------------------------------------------------
[INFO]  Installing or updating plugin "pypi:pybuilder_bandit, module name 'pybuilder_bandit'"
[INFO]  Processing plugin packages 'pybuilder_bandit' to be installed with {}
[INFO]  Activated environments: unit
[INFO]  Building lava-nc/lava-dl version 0.2.0
......  PyBuilder Logs ...
[INFO]  Running unit tests
[INFO]  Executing unit tests from Python modules in /home/user/lava-dl/tests
[INFO]  Executed 80 unit tests
[INFO]  All unit tests passed.
......  PyBuilder Logs ...
------------------------------------------------------------
BUILD SUCCESSFUL
------------------------------------------------------------
Build Summary
             Project: lava-nc/lava-dl
             Version: 0.2.0
      Base directory: /home/user/lava-dl
        Environments: unit
               Tasks: prepare [45089 ms] analyze [660 ms] compile_sources [0 ms] run_unit_tests [184641 ms] package [1086 ms] run_integration_tests [0 ms] verify [0 ms] publish [15128 ms]
Build finished at 2021-11-05 18:49:25
Build took 273 seconds (273800 ms)
```

### [Alternative] Installing Lava from Binaries

If you only need the lava package in your python environment, we will publish
Lava releases via
[GitHub Releases](https://github.com/lava-nc/lava-dl/releases). Please download
the package and install it.

Open a python terminal and run:

#### [Windows/MacOS/Linux]
```bash
$ python3 -m venv python3_venv
$ pip install -U pip
$ pip install lava-nc-0.1.0.tar.gz
```

## Getting Started

**End to end training tutorials**
* [Oxford spike train regression](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/oxford/train.ipynb)
* [MNIST digit classification](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/bootstrap/mnist/train.ipynb)
* [NMNIST digit classification](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/train.ipynb)
* [PilotNet steering angle prediction](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/pilotnet/train.ipynb)

**Deep dive training tutorials**
* [Dynamics and Neurons](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/neuron_dynamics/dynamics.ipynb)

**Inference tutorials**
* [Oxford Inference](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/netx/oxford/run.ipynb)
* [PilotNet SNN Inference](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/netx/pilotnet_snn/run.ipynb)
* [PilotNet SDNN Inference](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/netx/pilotnet_sdnn/run.ipynb)

## __`lava.lib.dl.slayer`__ 

`lava.lib.dl.slayer` is an enhanced version of [SLAYER](https://github.com/bamsumit/slayerPytorch). Most noteworthy enhancements are: support for _recurrent network structures_, a wider variety of _neuron models_ and _synaptic connections_ (a complete list of features is [here](https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/slayer/README.md)). This version of SLAYER is built on top of the [PyTorch](https://pytorch.org/) deep learning framework, similar to its predecessor. For smooth integration with Lava, `lava.lib.dl.slayer` supports exporting trained models using the platform independent __hdf5 network exchange__ format. 

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
        self.blocks = torch.nn.ModuleList([# sequential network blocks 
                slayer.block.sigma_delta.Input(sdnn_params), 
                slayer.block.sigma_delta.Conv(sdnn_params,  3, 24, 3),
                slayer.block.sigma_delta.Conv(sdnn_params, 24, 36, 3),
                slayer.block.rf_iz.Conv(rf_params, 36, 64, 3, delay=True),
                slayer.block.rf_iz.Conv(sdnn_cnn_params, 64, 64, 3, delay=True),
                slayer.block.rf_iz.Flatten(),
                slayer.block.alif.Dense(alif_params, 64*40, 100, delay=True),
                slayer.block.cuba.Recurrent(cuba_params, 100, 50),
                slayer.block.cuba.KWTA(cuba_params, 50, 50, num_winners=5)
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
assistant = slayer.utils.Assistant(net, error, optimizer, stats)
...
for epoch in range(epochs):
    for i, (input, ground_truth) in enumerate(train_loader):
        output = assistant.train(input, ground_truth)
        ...
    for i, (input, ground_truth) in enumerate(test_loader):
        output = assistant.test(input, ground_truth)
        ...
```
__Export the network__
```python
net.export_hdf5('network.net')
```

## __`lava.lib.dl.bootstrap`__

In general ANN-SNN conversion methods for rate based SNN result in high latency of the network during inference. This is because the rate interpretation of a spiking neuron using ReLU acitvation unit breaks down for short inference times. As a result, the network requires many time steps per sample to achieve adequate inference results.

`lava.lib.dl.bootstrap` enables rapid training of rate based SNNs by translating them to an equivalent dynamic ANN representation which leads to SNN performance close to the equivalent ANN and low latency inference. More details [here](https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/bootstrap/README.md). It also supports _hybrid training_ with a mixed ANN-SNN network to minimize the ANN to SNN performance gap. This method is independent of the SNN model being used.

It has similar API as `lava.lib.dl.slayer` and supports exporting trained models using the platform independent __hdf5 network exchange__ format.

### Example Code

__Import modules__
```python
import lava.lib.dl.bootstrap as bootstrap
```
__Network Description__
```python
# like any standard pyTorch network
class Network(torch.nn.Module):
    def __init__(self):
        ...
        self.blocks = torch.nn.ModuleList([# sequential network blocks 
                bootstrap.block.cuba.Input(sdnn_params), 
                bootstrap.block.cuba.Conv(sdnn_params,  3, 24, 3),
                bootstrap.block.cuba.Conv(sdnn_params, 24, 36, 3),
                bootstrap.block.cuba.Conv(rf_params, 36, 64, 3),
                bootstrap.block.cuba.Conv(sdnn_cnn_params, 64, 64, 3),
                bootstrap.block.cuba.Flatten(),
                bootstrap.block.cuba.Dense(alif_params, 64*40, 100),
                bootstrap.block.cuba.Dense(cuba_params, 100, 10),
            ])

    def forward(self, x, mode):
        ...
        for block, m in zip(self.blocks, mode):
            x = block(x, mode=m)

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
scheduler = bootstrap.routine.Scheduler()
...
for epoch in range(epochs):
    for i, (input, ground_truth) in enumerate(train_loader):
        mode = scheduler.mode(epoch, i, net.training)
        output = net.forward(input, mode)
        ...
        loss.backward()
    for i, (input, ground_truth) in enumerate(test_loader):
        mode = scheduler.mode(epoch, i, net.training)
        output = net.forward(input, mode)
        ...
```
__Export the network__
```python
net.export_hdf5('network.net')
```

## __`lava.lib.dl.netx`__ 

For inference using Lava, `lava.lib.dl.netx` provides an automated API for loading SLAYER-trained models as Lava Processes, which can be directly run on a desired backend. `lava.lib.dl.netx` imports models saved via SLAYER using the hdf5 network exchange format. The details of hdf5 network description specification can be found [here](https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/netx/README.md).

### Example Code

__Import modules__
```python
from lava.lib.dl.netx import hdf5
```
__Load the trained network__
```python
# Import the model as a Lava Process
net = hdf5.Network(net_config='network.net')
```
__Attach Processes for Input-Output interaction__
```python
from lava.proc import io

# Instantiate the processes
dataloader = io.dataloader.SpikeDataloader(dataset=test_set)
output_logger = io.sink.RingBuffer(shape=net.out_layer.shape, buffer=num_steps)
gt_logger = io.sink.RingBuffer(shape=(1,), buffer=num_steps)

# Connect the input to the network:
dataloader.ground_truth.connect(gt_logger.a_in)
dataloader.s_out.connect(net.in_layer.neuron.a_in)

# Connect network-output to the output process
net.out_layer.out.connect(output_logger.a_in)
```
__Run the network__
```python
from lava.magma import run_configs as rcfg
from lava.magma import run_conditions as rcnd

net.run(condition=rcnd.RunSteps(total_run_time), run_cfg=rcfg.Loihi1SimCfg())
output = output_logger.data.get()
gts = gt_logger.data.get()
net.stop()
```

