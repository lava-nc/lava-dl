import typing
import h5py
import pathlib
import nir
import numpy as np
import logging
from typing import Tuple

import torch

from lava.lib.dl import slayer


PATH_TYPE = typing.Union[str, pathlib.Path]


def read_neuron(layer: h5py.Group, shape: Tuple[int] = None) -> nir.NIRNode:
    """Reads a CUBA LIF layer from a h5py.Group.

    TODOs:
    - what if the layer is more than 1D?
    - handle scaleRho tauRho theta
    - support graded spikes
    - support refdelay
    - support other neuron types

    If neuron model not supported, warning logged and return None.
    """
    logging.debug(f"read_neuron {layer['neuron']['type'][()]}")

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
        tau_mem = 1. / float(vdecay) if vdecay != 0 else np.inf
        tau_syn = 1. / float(idecay) if idecay != 0 else np.inf

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

    # if there is no input layer, then create one
    first_layer = network[str(layer_keys[0])]
    if first_layer['type'][0] != b'input':
        logging.info('--- INPUT')
        logging.warning('no input layer found, creating one')
        if first_layer['type'][0] != b'dense':
            raise ValueError('first layer must be input or dense')
        current_shape = first_layer['weight'][:].shape[1:]
        nodes.append(nir.Input(shape=current_shape))
        edges.append((len(nodes) - 1, len(nodes)))

    # iterate over layers
    for layer_idx_int in layer_keys:
        lidx = str(layer_idx_int)
        layer = network[lidx]

        logging.info(f"--- Layer #{lidx}: {layer['type'][0].decode().upper()}")
        logging.debug(f'current shape: {current_shape}')

        if layer['type'][0] == b'dense':
            # shape, type, neuron, inFeatures, outFeatures, weight, (delay)
            logging.debug(f'dense weights of shape {layer["weight"][:].shape}')

            # make sure weight matrix matches shape of previous layer
            if current_shape is None:
                if len(layer['weight'][:].shape) != 2:
                    raise AssertionError('shape mismatch')
                current_shape = layer['weight'][:].shape[1]
            elif isinstance(current_shape, int):
                if current_shape != layer['weight'][:].shape[-1]:
                    raise AssertionError('shape mismatch')
            else:
                if len(current_shape) != 1:
                    raise AssertionError('shape mismatch')
                if current_shape[0] != layer['weight'][:].shape[1]:
                    raise AssertionError('shape mismatch')

            # infer shape of current layer
            if len(layer['weight'][:].shape) not in [1, 2]:
                raise AssertionError('invalid dim')
            is_1d = len(layer['weight'][:].shape) == 1
            current_shape = 1 if is_1d else layer['weight'][:].shape[0]

            # store the weight matrix (np.array, carrying over type)
            if 'bias' in layer:
                w = layer['weight'][:]
                b = layer['bias'][:]
                nodes.append(nir.Affine(weight=w, bias=b))
            else:
                nodes.append(nir.Linear(weight=layer['weight'][:]))

            # store the neuron group
            neuron = read_neuron(layer)
            if neuron is None:
                raise NotImplementedError('could not read neuron')
            nodes.append(neuron)

            # connect linear to neuron, neuron to next element
            edges.append((len(nodes) - 2, len(nodes) - 1))
            edges.append((len(nodes) - 1, len(nodes)))

        elif layer['type'][0] == b'input':
            # iDecay, refDelay, scaleRho, tauRho, theta
            # type, vDecay, vThMant, wgtExp
            current_shape = layer['shape'][:]
            nodes.append(nir.Input(shape=layer['shape'][:]))
            edges.append((len(nodes) - 1, len(nodes)))
            logging.debug(f'keys: {layer.keys()}')
            w = layer['weight'][()]
            b = layer['bias'][()]
            logging.debug(f'shape: {layer["shape"][:]}, bias: {b}, weight: {w}')
            n_keys = list(layer['neuron'].keys())
            logging.debug(f'neuron keys: {", ".join(n_keys)}')

        elif layer['type'][0] == b'flatten':
            # shape, type
            flattened_shape = layer['shape'][:]
            logging.debug(f"flattening -> {flattened_shape}")

            if len(nodes) == 0:
                raise AssertionError('must be preceded by a layer')
            if not isinstance(current_shape, tuple):
                raise AssertionError('nothing to flatten')
            if len(current_shape) <= 1:
                raise AssertionError('nothing to flatten')
            if len(current_shape) < len(flattened_shape):
                raise AssertionError('shape mismatch')
            if np.prod(current_shape) != np.prod(flattened_shape):
                raise AssertionError('shape mismatch')

            if len(current_shape) == len(flattened_shape):
                # (A, B, C) -> (1, 1, A*B*C)
                axes_to_flatten = []
                for i in range(len(current_shape)):
                    if current_shape[i] != 1 and flattened_shape[i] == 1:
                        axes_to_flatten.append(i)
                # check if dims to flatten are next to each other
                if not np.alltrue(np.diff(axes_to_flatten) == 1):
                    raise AssertionError('dims to flatten are not contiguous')
                nodes.append(nir.Flatten(
                    start_dim=axes_to_flatten[0],
                    end_dim=axes_to_flatten[1]
                ))
            else:
                # (A, B, C) -> (A*B*C)
                # assume dimensions to be flattened are next to each other
                start_flatten = None
                for i in range(len(current_shape)):
                    if current_shape[i] != flattened_shape[i]:
                        start_flatten = i
                        break
                if start_flatten is None:
                    logging.warning('nothing to flatten')
                    continue
                end_flatten = -1
                for i in range(start_flatten, len(current_shape)):
                    exp_dim = np.prod(current_shape[start_flatten:i + 1])
                    cur_dim = flattened_shape[start_flatten]
                    if exp_dim == cur_dim:
                        end_flatten = i - 1
                        break
                nodes.append(nir.Flatten(
                    start_dim=start_flatten, end_dim=end_flatten
                ))

            current_shape = int(np.prod(current_shape))
            edges.append((len(nodes) - 1, len(nodes)))

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
            logging.debug(f'strd {stride} pad {pad} dil {dil} w {weight.shape}')

            # infer shape of current layer
            if in_channels != current_shape[0]:
                raise AssertionError('in_channels mismatch')
            x_prev = current_shape[1]
            y_prev = current_shape[2]
            nomin = (x_prev + 2 * pad[0] - dil[0] * (kernel_size[0] - 1) - 1)
            x = nomin // stride[0] + 1
            nomin = (y_prev + 2 * pad[1] - dil[1] * (kernel_size[1] - 1) - 1)
            y = nomin // stride[1] + 1
            current_shape = (out_channels, x, y)

            # check for unsupported options
            if layer['groups'][()] != 1:
                logging.warning('groups not supported, setting to 1')
            if 'delay' in layer:
                delay = layer['delay'][()]
                logging.warning(f"delay={delay} unsupported, ignore")

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
            neuron = read_neuron(layer)
            if neuron is None:
                raise NotImplementedError('could not read neuron')
            nodes.append(neuron)

            # connect conv to neuron group, neuron group to next element
            edges.append((len(nodes) - 2, len(nodes) - 1))
            edges.append((len(nodes) - 1, len(nodes)))

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

    return nir.NIRGraph(
        nodes={str(i): nodes[i] for i in range(len(nodes))},
        edges=edges
    )


def convert_to_nir(net_config: PATH_TYPE,
                   output_path: PATH_TYPE) -> nir.NIRGraph:
    """Load a NIR from a HDF/conn5 netx file."""
    with h5py.File(net_config, "r") as f:
        nir_graph = read_node(f["layer"])
    nir.write(output_path, nir_graph)


########################################################################
# NIR -> LAVA
########################################################################


class NetworkNIR(torch.nn.Module):
    def __init__(self, blocks):
        super(NetworkNIR, self).__init__()
        self.blocks = torch.nn.ModuleList(blocks)

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


def nir_to_lavadl_net(graph: nir.NIRGraph) -> NetworkNIR:
    """Converts a NIRGraph to a Lava-dl network."""
    nodes = graph.nodes
    edges = graph.edges

    # make sure that the graph is acyclic
    visited = set()
    for edge in edges:
        visited.add(edge[0])
        if edge[1] in visited:
            raise ValueError("Lava does not support cyclic graphs")

    # get input node key
    input_node_keys = [k for k in nodes if isinstance(nodes[k], nir.Input)]
    logging.debug(f'input_node_keys: {input_node_keys}')
    if len(input_node_keys) > 1:
        raise AssertionError("do not support multiple input nodes")
    if len(input_node_keys) == 0:
        # get the first node - remove every node that has a predecessor
        input_node_keys = list(nodes.keys())
        for edge in edges:
            if str(edge[1]) in input_node_keys:
                input_node_keys.remove(str(edge[1]))
        if len(input_node_keys) != 1:
            raise AssertionError("could not find single input node")
    node_key = input_node_keys[0]
    logging.debug(f'input node key: {node_key}')

    # make sure that the graph is connected
    visited = set()
    to_visit = set([int(node_key)])
    while len(to_visit) > 0:
        nk = to_visit.pop()
        visited.add(nk)
        for edge in edges:
            if edge[0] == nk:
                to_visit.add(edge[1])
    if len(visited) != len(nodes):
        logging.debug(f"visited: {visited}, nodes: {nodes}")
        raise ValueError("Lava does not support disconnected graphs")

    # make sure the graph doesn't split or join
    # TODO: we should support skip connections in lava-dl?
    for edge in edges:
        for edge_ in edges:
            if edge_[0] == edge[0] and edge_[1] != edge[1]:
                raise ValueError("Lava does not support graphs with splits")
            elif edge_[0] != edge[0] and edge_[1] == edge[1]:
                raise ValueError("Lava does not support graphs with joins")

    # create the network
    blocks = []
    input_shape = None

    def get_next_node(node_key):
        """Returns next node key in graph, or None if there is no next node."""
        next_node_keys = [str(e[1]) for e in edges if str(e[0]) == node_key]
        if len(next_node_keys) > 1:
            raise AssertionError("currently do not support branching")
        return None if len(next_node_keys) == 0 else next_node_keys[0]

    while node_key is not None:
        node = nodes[node_key]

        logging.info(f"--- Layer #{node_key}: {type(node).__name__}")

        if isinstance(node, nir.Input):
            # TODO: check neuron model in network (here: assume CUBA)
            logging.debug(f'input of shape: {node.shape}')
            blocks.append(slayer.block.cuba.Input({
                'threshold'     : 0.1,
                'current_decay' : 1,
                'voltage_decay' : 0.1,
            }))
            # store input shape to later infer all shapes
            input_shape = node.shape

        elif isinstance(node, nir.Flatten):
            # TODO: check what shape is expected (start_dim, end_dim)
            blocks.append(slayer.block.cuba.Flatten())

        elif isinstance(node, nir.Conv2d):
            node_key = get_next_node(node_key)
            next_node = nodes[node_key]
            if not isinstance(next_node, nir.CubaLIF):
                raise AssertionError("only supports CUBA")
            # neuron parameters
            if next_node.tau_syn == np.inf:
                logging.warning('tau_syn is inf, setting to 0')
                i_decay = 0
            else:
                i_decay = 1. / next_node.tau_syn
            if next_node.tau_mem == np.inf:
                logging.warning('tau_mem is inf, setting to 0')
                v_decay = 0
            else:
                v_decay = 1. / next_node.tau_mem
            threshold = next_node.v_threshold
            neuron_params = {
                'threshold'     : threshold,
                'current_decay' : i_decay,
                'voltage_decay' : v_decay,
            }
            # conv block parameters
            logging.debug(f'weights of shape: {node.weight.shape}')
            if len(node.weight.shape) != 4:
                raise AssertionError("only support Conv2D")
            conv_block = slayer.block.cuba.Conv(
                neuron_params=neuron_params,
                in_features=node.weight.shape[1],
                out_features=node.weight.shape[0],
                kernel_size=node.weight.shape[2:],
                stride=node.stride,
                padding=node.padding,
                dilation=node.dilation,
                groups=node.groups
            )
            blocks.append(conv_block)

        elif isinstance(node, nir.Linear) or isinstance(node, nir.Affine):
            node_key = get_next_node(node_key)
            next_node = nodes[node_key]
            if not isinstance(next_node, nir.CubaLIF):
                raise AssertionError("only support Linear-CUBA")
            # neuron parameters
            if next_node.tau_syn == np.inf:
                logging.warning('tau_syn is inf, setting to 0')
                i_decay = 0
            else:
                i_decay = 1. / next_node.tau_syn
            if next_node.tau_mem == np.inf:
                logging.warning('tau_mem is inf, setting to 0')
                v_decay = 0
            else:
                v_decay = 1. / next_node.tau_mem
            threshold = next_node.v_threshold
            neuron_params = {
                'threshold'     : threshold,
                'current_decay' : i_decay,
                'voltage_decay' : v_decay,
            }
            # linear block parameters
            logging.debug(f'weights of shape: {node.weight.shape}')
            if len(node.weight.shape) > 2:
                raise AssertionError("only support 2D Linear")
            is_1d = len(node.weight.shape) == 1
            in_neurons = node.weight.shape[0] if is_1d else node.weight.shape[1]
            out_neurons = 1 if is_1d else node.weight.shape[0]
            linear_block = slayer.block.cuba.Dense(
                neuron_params=neuron_params,
                in_neurons=in_neurons,
                out_neurons=out_neurons,
            )
            blocks.append(linear_block)

        else:
            raise ValueError(f"Unsupported node type {type(node)}")

        node_key = get_next_node(node_key)

    # create the network
    network = NetworkNIR(blocks=blocks)

    # pass example data (B, *input_shape, T) through network to infer shapes
    data_shape = (32, *input_shape, 16)
    logging.debug(f'input_shape: {input_shape}, data_shape: {data_shape}')
    network.forward(torch.zeros(data_shape))

    return network
