import typing
import h5py
import pathlib
import nir
import numpy as np
import logging
from typing import Tuple


PATH_TYPE = typing.Union[str, pathlib.Path]


def read_cuba_lif(layer: h5py.Group, shape: Tuple[int] = None) -> nir.NIRNode:
    """Reads a CUBA LIF layer from a h5py.Group.

    TODOs: 
    - what if the layer is more than 1D?
    - handle scaleRho tauRho theta
    - support graded spikes
    - support refdelay
    - support other neuron types

    If the neuron model is not supported, a warning is logged and None is returned.
    """
    logging.debug(f"read_cuba_lif {layer['neuron']['type'][()]}")

    if 'gradedSpike' in layer['neuron']:
        if layer['neuron']['gradedSpike'][()]:
            logging.warning('graded spikes not supported')

    if layer['neuron']['type'][()] in [b'LOIHI', b'CUBA']:
        if layer['neuron']['refDelay'][()] != 1:
            logging.warning('refdelay not supported, setting to 1')
        if layer['neuron']['vDecay'] == 0:
            logging.warning('vDecay is 0, setting to inf')
        if layer['neuron']['iDecay'] == 0:
            logging.warning('iDecay is 0, setting to inf')

        vdecay = layer['neuron']['vDecay'][()]
        idecay = layer['neuron']['iDecay'][()]
        tau_mem = 1./float(vdecay) if vdecay != 0 else np.inf
        tau_syn = 1./float(idecay) if idecay != 0 else np.inf

        return nir.CubaLIF(
            tau_syn=tau_syn,
            tau_mem=tau_mem,
            r=1.,  # no scaling of synaptic current
            v_leak=0.,  # currently no bias in Loihi's neurons
            v_threshold=layer['neuron']['vThMant'][()],
            w_in=tau_syn  # w_in_eff = win / tau_syn
        )
    else:
        logging.warning('currently only support for CUBA-LIF')
        logging.error(f"no support for {layer['neuron']['type'][()]}")
        return None


def read_node(network: h5py.Group) -> nir.NIRNode:
    """Read a graph from a HDF/conn5 file.
    
    TODOs:
    - support delay in convolutional layers
    """
    nodes = []
    edges = []
    current_shape = None

    # need to sort keys as integers, otherwise does 1->10->2
    layer_keys = sorted(list(map(int, network.keys())))

    # iterate over layers
    for layer_idx_int in layer_keys:
        layer_idx = str(layer_idx_int)
        layer = network[layer_idx]

        logging.info(f"--- Layer #{layer_idx}: {layer['type'][0].decode().upper()}")
        logging.debug(f'current shape: {current_shape}')

        if layer['type'][0] == b'dense':
            # shape, type, neuron, inFeatures, outFeatures, weight, delay?
            logging.debug(f'dense weights of shape {layer["weight"][:].shape}')

            # make sure weight matrix matches shape of previous layer
            if current_shape is None:
                assert len(layer['weight'][:].shape) == 2, 'shape mismatch in dense'
                current_shape = layer['weight'][:].shape[1]
            elif isinstance(current_shape, int):
                assert current_shape == layer['weight'][:].shape[-1], 'shape mismatch in dense'
            else:
                assert len(current_shape) == 1, 'shape mismatch in dense'
                assert current_shape[0] == layer['weight'][:].shape[1], 'shape mismatch in dense'

            # infer shape of current layer
            assert len(layer['weight'][:].shape) in [1, 2], 'invalid dimension for dense layer'
            current_shape = 1 if len(layer['weight'][:].shape) == 1 else layer['weight'][:].shape[0]

            # store the weight matrix (np.array, carrying over type)
            if 'bias' in layer:
                nodes.append(nir.Affine(weight=layer['weight'][:], bias=layer['bias'][:]))
            else:
                nodes.append(nir.Linear(weight=layer['weight'][:]))

            # store the neuron group
            neuron = read_cuba_lif(layer)
            if neuron is None:
                raise NotImplementedError('could not read neuron')
            nodes.append(neuron)

            # connect linear to neuron, neuron to next element
            edges.append((len(nodes)-2, len(nodes)-1))
            edges.append((len(nodes)-1, len(nodes)))

        elif layer['type'][0] == b'input':
            # iDecay, refDelay, scaleRho, tauRho, theta, type, vDecay, vThMant, wgtExp
            current_shape = layer['shape'][:]
            logging.warning('INPUT - not implemented yet')
            logging.debug(f'keys: {layer.keys()}')
            logging.debug(f'shape: {layer["shape"][:]}, bias: {layer["bias"][()]}, weight: {layer["weight"][()]}')
            logging.debug(f'neuron keys: {", ".join(list(layer["neuron"].keys()))}')

        elif layer['type'][0] == b'flatten':
            # shape, type
            logging.debug(f"flattening shape (ignored): {layer['shape'][:]}")
            # check last layer's size
            assert len(nodes) > 0, 'flatten layer: must be preceded by a layer'
            assert isinstance(current_shape, tuple), 'flatten layer: nothing to flatten'
            last_node = nodes[-1]
            nodes.append(nir.Flatten(n_dims=1))
            current_shape = int(np.prod(current_shape))
            edges.append((len(nodes)-1, len(nodes)))

        elif layer['type'][0] == b'conv':
            # shape, type, neuron, inChannels, outChannels, kernelSize, stride, 
            # padding, dilation, groups, weight, delay?
            weight = layer['weight'][:]
            stride = layer['stride'][()]
            pad = layer['padding'][()]
            dil = layer['dilation'][()]
            kernel_size = layer['kernelSize'][()]
            in_channels = layer['inChannels'][()]
            out_channels = layer['outChannels'][()]
            logging.debug(f'stride {stride} padding {pad} dilation {dil} w {weight.shape}')

            # infer shape of current layer
            assert in_channels == current_shape[0], 'in_channels must match previous layer'
            x_prev = current_shape[1]
            y_prev = current_shape[2]
            x = (x_prev + 2*pad[0] - dil[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
            y = (y_prev + 2*pad[1] - dil[1]*(kernel_size[1]-1) - 1) // stride[1] + 1
            current_shape = (out_channels, x, y)

            # check for unsupported options
            if layer['groups'][()] != 1:
                logging.warning('groups not supported, setting to 1')
            if 'delay' in layer:
                logging.warning(f"delay=({layer['delay'][()]}) not supported, ignoring")

            # store the conv matrix (np.array, carrying over type)
            nodes.append(nir.Conv2d(
                weight=layer['weight'][:],
                bias=layer['bias'][:] if 'bias' in layer else None,
                stride=stride,
                padding=pad,
                dilation=dil,
                groups=layer['groups'][()]
            ))

            # store the neuron group
            neuron = read_cuba_lif(layer)
            if neuron is None:
                raise NotImplementedError('could not read neuron')
            nodes.append(neuron)

            # connect conv to neuron group, neuron group to next element
            edges.append((len(nodes)-2, len(nodes)-1))
            edges.append((len(nodes)-1, len(nodes)))

        elif layer['type'][0] == b'average':
            # shape, type
            logging.error('AVERAGE LAYER - not implemented yet')
            raise NotImplementedError('average layer not implemented yet')

        elif layer['type'][0] == b'concat':
            # shape, type, layers
            logging.error('CONCAT LAYER - not implemented yet')
            raise NotImplementedError('concat layer not implemented yet')

        elif layer['type'][0] == b'pool':
            # shape, type, neuron, kernelSize, stride, padding, dilation, weight
            logging.error('POOL LAYER - not implemented yet')
            raise NotImplementedError('pool layer not implemented yet')

        else:
            logging.error('layer type not supported:', layer['type'][0])

    # remove last edge (no next element)
    edges.pop(-1)

    return nir.NIRGraph(nodes=nodes, edges=edges)


def convert_to_nir(net_config: PATH_TYPE, path: PATH_TYPE) -> nir.NIRGraph:
    """Load a NIR from a HDF/conn5 file."""
    with h5py.File(net_config, "r") as f:
        nir_graph = read_node(f["layer"])
    nir.write(path, nir_graph)


class Network:
    def __init__(self, path: typing.Union[str, pathlib.Path]) -> None:
        nir_graph = nir.read(path)
        self.graph = nir_graph
        # TODO: implement the NIR -> Lava conversion
        pass
