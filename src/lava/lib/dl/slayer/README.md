# Lava-DL SLAYER

`lava.lib.dl.slayer` is an enhanced version of [SLAYER](https://github.com/bamsumit/slayerPytorch). It now supports a wide variety of learnable event-based _neuron models_, _synapse_, _axon_, and _dendrite_ properties. Other enhancements include various utilities useful during training for event IO, visualization,and filtering as well as logging of training statistics. 

**Highlight Features**

* Resonator, Adaptive leaky neuron dynamics in addtion to conventional Leaky neuron dynamics
* Sigma-Delta wrapper around arbitrary neuron dynamics
* Graded spikes
* Learnable neuron parameters at a granularity of individual neuron
* Persistent states between iterations for robotics application
* Arbitrary recurrent architectures including k-winner-take-all (KWTA)
* Complex valued synapses
* Sparse connectivity with connection masking
* Runtime shape identification (eliminates the need for _a priori_ architecture shape calculation)
* Just-In-Time compilation of CUDA acccelerated code.
* Block interface for easy description of network.
* Easy network export to hdf5 interface format.

## Tutorials

**End to End**
* [Oxford spike train regression](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/oxford/train.ipynb)
* [NMNIST digit classification](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/train.ipynb)
* [PilotNet steering angle prediction](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/pilotnet/train.ipynb)

**Deep Dive**
* [Dynamics and Neurons](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/neuron_dynamics/dynamics.ipynb)


## Modules

The overall feature organization is described below.

### Spike (`slayer.spike`)
SLAYER supports binary as well as graded spikes, which are amenable to backpropagation. This opens the door for a new class of neuron behavior.

### Neuron (`slayer.neuron`)
Neuron models in SLAYER are built around custom CUDA accelerated fundamental linear dynamics. Each neuron model has individually learnable parameters from its neural dynamics as well as persistent state behavior between iterations. The following neuron dynamics are supported.
* Leaky Integrator
* Resonator
* Adaptive Integrator with Refractory Dynamics
* Adaptive Resonator with Refractory Dynamics

These fundamental dynamics can be combined to build a variety of neuron models. Following neuron models are currently supported:

#### CUrrent BAsed leaky integrator: `slayer.neuron.cuba`
<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/135405316-0782e174-ceaf-4d97-a4ca-7ddcd681a1ba.png" alt="Drawing" style="width=1000px"/>
</p>

#### Adaptive Leaky Integrate and Fire: `slayer.neuron.alif`
<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/135405926-60269c92-92d7-453c-8324-941b3322c7a5.png" alt="Drawing" style="width=1000px"/>
</p>

#### Resonate and Fire (phase threshold and Izhikevich variant): `slayer.neuron.{rf, rf_iz}`
<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/135404915-3e9371c4-3148-4ea8-813e-8f05ce9e4b67.png" alt="Drawing" style="width=1000px"/>
</p>
    
#### Adaptive resonators: `slayer.neuron.{adrf, adrf_iz}`
<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/135405837-03c1a053-03fc-44bf-afe1-2cdadde4f01a.png" alt="Drawing" style="width=1000px"/>
</p>

#### Sigma Delta neuron with arbitrary activation: `slayer.neuron.sigma_delta`
<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/135405757-0747aae0-def6-49cd-aa44-8b0fa67b40fd.png" alt="Drawing" style="width=1000px"/>
</p>

In addition, SLAYER also supports _neuron dropout_ and quantization ready batch-normalization methods.

### Synapse (`slayer.syanpse`)

SLAYER supports dense, conv, and pool synaptic connections. Masking is possible in both real as well as complex connections: `slayer.synapse.{complex}.{Dense, Conv, Pool}`.

### Axon (`slayer.axon`)

* Learnable axonal delay (`slayer.axon.Delay`)
* Learnable delta encoder (`slayer.axon.Delta`)

### Dendrite (`slayer.dendrite`)

* Sigma decoder (`slayer.dendrite.Sigma`)

### Blocks (`slayer.blocks`)

SLAYER provides easy encapsulation of neuron, synapse, axon, and dendrite classes for a variety of standard neuron-connection combinations:
`slayer.block.{cuba, alif, rf, rf_iz, sigma_delta}.{input, output, dense, conv, pool, kwta, recurrent}`
These blocks can be easily used to define a network and export it in pytorch as well as our platform independent hdf5 format.

```python
class Network(torch.nn.Module):
    def __init__(self):
        ...
        self.blocks = torch.nn.ModuleList(
            [  # sequential network blocks
                slayer.block.sigma_delta.Input(sdnn_params),
                slayer.block.sigma_delta.Conv(sdnn_params, 3, 24, 3),
                slayer.block.sigma_delta.Conv(sdnn_params, 24, 36, 3),
                slayer.block.rf_iz.Conv(rf_params, 36, 64, 3, delay=True),
                slayer.block.rf_iz.Conv(sdnn_cnn_params, 64, 64, 3, delay=True),
                slayer.block.rf_iz.Flatten(),
                slayer.block.alif.Dense(alif_params, 64 * 40, 100, delay=True),
                slayer.block.cuba.Recurrent(cuba_params, 100, 50),
                slayer.block.cuba.KWTA(cuba_params, 50, 50, num_winners=5),
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
        return x

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, "w")
        layer = h.create_group("layer")
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f"{i}"))
```
<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/135402787-ca849ef2-697d-4c5c-9f05-9b6fe3c3b072.png" alt="Drawing" style="height: 400px;"/>
</p>

### Fundamental Practices

* Tensors are always assumed to be in the order `NCHWT` or `NCT` where `N`:Batch, `C`:Channel, `H`: Height(y), `W`: Width(x) and `T`: Time. 
    * `NCHW` is the default PyTorch ordering. 
* Synapse values are maintained in scaled down range.
* Neurons hold the shape of the layer. It shall be automatically identified on runtime.
