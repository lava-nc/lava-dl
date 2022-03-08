# Lava-dl-netx

`lava.lib.dl.netx` automates deep learning network model exchange to/from LAVA from/to other frameworks. At the moment, we support a simple platform independent hdf5 network description protocol. In furture we will extend network exchange support to other neural network exchange formats. 

Loading a model to Lava is as simple as:
```python
from lava.lib.dl.netx import hdf5
# Import the model as a Lava Process
net = hdf5.Network(net_config='network.net')
```
<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/135401882-12433d6e-b38e-488f-be2f-1aa3a3a14fda.png" alt="Drawing" style="height: 400px;"/>
</p>
The hdf5 network description protocol is described below:

## HDF5 description protocol
* The computational graph is represented layer-wise in the `layer` field of the hdf5 file.
* The layer id is assigned serially from `0` to `n-1` starting from input to output.
    * By default, a sequential connection is assumed.
    * Skip/recurrent connections are preceded by a concatenate layer that connects to pervious layer plus a list of non-sequential layers identified by their id.
    * Each layer entry consts a minimum of `shape` and `type` field. Other relevant fileds can be addes as necessary.
        * `shape` entry is a tuple/list in (x, y, z) format.
        * `type` entry is a string that describes the layer type. See below for a list of supported types.
        * `neuron` field describes the compartment model and it's parameter.
            * default `neuron` type is `CUBA-LIF`.
```
|
|->layer # description of network layer blocks such as input, dense, conv, pool, flatten, average
|   |->0
|   |  |->{shape, type, ...} # each layer description has at least shape and type attribute
|   |->1
|   |  |->{shape, type, ...}
|   :
|   |->n-1
|      |->{shape, type, ...}
| 
| # other fields (not used for network exchange)
|->simulation # simulation description
|   |->Ts # sampling time. Usually 1
|   |->tSample # length of the sample to run
```

### Supported layer types
```
input  : {shape, type}
flatten: {shape, type}
average: {shape, type}
concat : {shape, type, layers}
dense  : {shape, type, neuron, inFeatures, outFeatures, weight, delay(if available)}
pool   : {shape, type, neuron, kernelSize, stride, padding, dilation, weight}
conv   : {shape, type, neuron, inChannels, outChannels, kernelSize, stride,
                        |      padding, dilation, groups, weight, delay(if available)}
                        |
                        |-> this is the description of the compartment parameters
                        |-> {iDecay, vDecay, vThMant, refDelay, ... (other additional params)}
```
