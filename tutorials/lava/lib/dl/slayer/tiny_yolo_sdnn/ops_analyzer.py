from datetime import datetime

import numpy as np
import torch


def compare_ops(blocks, counts, input_shape, handle=None, skip_l0=True):
    shapes = [input_shape] + [b.shape for b in blocks if hasattr(b, 'neuron')]

    # synops calculation
    sdnn_synops = []
    ann_synops = []
    syn_valid = []
    syn_elems = []
    for l in range(len(blocks)):
        if hasattr(blocks[l], 'neuron') is False:
            break
        
        syn_valid.append(float(torch.sum(blocks[l].synapse.pre_hook_fx(blocks[l].synapse.weight) != 0)))
        syn_elems.append(float(blocks[l].synapse.weight.nelement()))

        conv_synops = ( # ignoring padding
                counts[l] * np.prod(shapes[l])
                * blocks[l].synapse.out_channels
                * np.prod(blocks[l].synapse.kernel_size)
                / np.prod(blocks[l].synapse.stride)
            )
        sdnn_synops.append(conv_synops)
        ann_synops.append(conv_synops / counts[l])

    # event and synops comparison
    total_events = np.sum([c*np.prod(s) for c, s in zip(counts, shapes)])
    total_synops = np.sum(sdnn_synops)
    total_ann_activs = np.sum([np.prod(s) for s in shapes])
    total_ann_synops = np.sum(ann_synops)
    total_valid = np.sum(syn_valid)
    total_elems = np.sum(syn_elems)

    table = []
    table.append(f'|{"-"*88}|')
    table.append(f'| {" "*24} |          SDNN            |           ANN           |        |')
    table.append(f'|{"-"*88}|')
    table.append(f'| {" "*7} |     Shape      |  Events   |    Synops    | Activations|    MACs    | prune  |')
    table.append(f'|{"-"*88}|')
    for l in range(min(len(counts), len(shapes))):
        if l==0 and skip_l0==True:
            continue
        row = ''
        row += f'| layer-{l} | '
        if len(shapes[l]) == 3: z, y, x = shapes[l]
        elif len(shapes[l]) == 1:
            z = shapes[l][0]
            y = x = 1
        row += f'({x:-3d},{y:-3d},{z:-4d}) | {counts[l] * np.prod(shapes[l]):9.2f} | '
        if l==0:
            row += f'{" "*12} | {np.prod(shapes[l]):-10.0f} | {" "*10} | {" "*6} |'
        else:
            row += f'{sdnn_synops[l-1]:12.2f} | {np.prod(shapes[l]):10.0f} | {ann_synops[l-1]:10.0f} | {(syn_elems[l-1] + 1e-6) / (syn_valid[l-1] + 1e-6):6.3f}x|'
        table.append(row)
    table.append(f'|{"-"*88}|')
    table.append(f'|  Total  | {" "*14} | {total_events:9.2f} | {total_synops:12.2f} | {total_ann_activs:10.0f} | {total_ann_synops:10.0f} | {total_elems / total_valid:6.3f}x|')
    table.append(f'|{"-"*88}|')
    
    for row in table:
        print(row)
    
    if handle is not None:
        for row in table:
            handle.write(row + '\n')

    return np.array([total_events, total_synops, total_ann_activs, total_ann_synops, total_valid, total_elems])

def analyze_ops(net, all_counts, filename=None):
    if filename is not None:
        handle = open(filename, 'wt')
    else:
        handle = None

    counts = np.mean(all_counts, axis=0).flatten()
    offset = 0
    ops = 0

    input_counts = counts[offset:offset + len(net.input_blocks)]
    offset += len(net.input_blocks)
    # compare_ops(net.input_blocks, input_counts)

    backend_counts = [input_counts[-1], *counts[offset:offset + len(net.backend_blocks)]]
    offset += len(net.backend_blocks)
    ops += compare_ops(net.backend_blocks, backend_counts, net.input_blocks[-1].neuron.shape, handle, skip_l0=False)

    h1_backend_counts = [backend_counts[-1], *counts[offset:offset + len(net.head1_backend)]]
    offset += len(net.head1_backend)
    ops += compare_ops(net.head1_backend, h1_backend_counts, net.backend_blocks[-1].neuron.shape, handle)
    
    h1_blocks_counts = [h1_backend_counts[-1], *counts[offset:offset + len(net.head1_blocks)]]
    offset += len(net.head1_blocks)
    ops += compare_ops(net.head1_blocks, h1_blocks_counts, net.head1_backend[-1].neuron.shape, handle)
    
    h2_backend_counts = [h1_backend_counts[-1], *counts[offset:offset + len(net.head2_backend)]]
    offset += len(net.head2_backend)
    ops += compare_ops(net.head2_backend, h2_backend_counts, net.backend_blocks[-1].neuron.shape, handle)
    
    h2_inp_counts = (h2_backend_counts[-1] + backend_counts[-1]) / 2
    h2_backend_shape = net.head2_backend[-1].neuron.shape
    backend_shape = net.backend_blocks[-1].neuron.shape
    h2_inp_shape = (h2_backend_shape[0] + backend_shape[0], *h2_backend_shape[1:])
    h2_blocks_counts = [h2_inp_counts, *counts[offset:offset + len(net.head2_blocks)]]
    offset += len(net.head2_blocks)
    ops += compare_ops(net.head2_blocks, h2_blocks_counts, h2_inp_shape, handle)

    total_events, total_synops, total_ann_activs, total_ann_synops, total_valid, total_elems = ops

    lines = []
    lines.append(f'Events sparsity: {total_ann_activs / total_events:5.2f}x')
    lines.append(f'Synops sparsity: {total_ann_synops / total_synops:5.2f}x')
    lines.append(f'Struct sparsity: {total_elems / total_valid:5.2f}x')
    lines.append('')
    lines.append(f'{total_events = :.2f}')
    lines.append(f'{total_synops = :.2f}')
    
    for l in lines:
        print(l)
        
    if filename is not None:
        for row in lines:
            handle.write(row + '\n')
    
    if handle is not None:
        handle.close()

    return ops
