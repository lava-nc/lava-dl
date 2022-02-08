import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io

from lava.lib.dl import netx
from dataset import PilotNetDataset

net = netx.hdf5.Network(net_config='conv_only_1chip.net', has_graded_input=True)
print(net)

for l in net.layers:
    print(l.has_graded_input)

full_set = PilotNetDataset(
    path='../data',
    size=[100, 33],
    transform=net.in_layer.transform, # input transform
    visualize=True, # visualize ensures the images are returned in sequence
)
train_set = PilotNetDataset(
    path='../data',
    size=[100, 33],
    transform=net.in_layer.transform, # input transform
    train=True,
)
test_set = PilotNetDataset(
    path='../data',
    size=[100, 33],
    transform=net.in_layer.transform, # input transform
    train=False,
)

num_samples = 200
steps_per_sample = 1
run_config = Loihi1SimCfg(select_tag='fixed_pt')

id = 10550
gts = []
results = []
for i in range(num_samples + len(net.layers)):
    image, gt = full_set[id]
    if i > 0:
        net.in_layer.neuron.bias.set(image)
        # assert np.abs(net.in_layer.neuron.bias.get() - image).sum() == 0
    else:
        net.in_layer.neuron.bias.init = image
    gts.append(gt)
    id = id + 1

    # run the network
    net.run(condition=RunSteps(num_steps=1), run_cfg=run_config)

    # gather result
    results.append(net.out_layer.neuron.sigma.get())
    print(f'\rSample: {id}', end='')
    # results.append(net.out_layer.neuron.v.get())
bias = net.in_layer.neuron.bias.get()
net.stop()
