import os
import sys
import numpy as np
from glob2 import glob
import time
from tqdm import tqdm
from numba import jit

@jit(parallel=True)
def process(events):
     
    imgs = []
    img = np.zeros([1, 346, 260])
    t_last = None
    dt = 1000000 / 150
    tau = dt * 5
    for e in events:
        if not t_last:
            t_last = e['t']
            t_next = e['t'] + dt
    
        img[0, e['x'], e['y']] += e['p']#np.exp(-(t_next - e[2]) / tau) * e['p']
    
        if e[2] - t_last >= dt:
            imgs.append(np.repeat(img.copy(), 3, 0))
            t_next += dt
            img *= np.exp(-(dt / tau))
            t_last = e['t']
 
root_dir = sys.argv[1]

fns = glob(root_dir + "/**/*.npz")

t0 = time.time()
for fn in tqdm(fns[:3]):
    data = np.load(fn, allow_pickle=True)
    events = np.zeros(len(data['t']), dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]))
    events['t'] = data['t']
    events['x'] = data['x']
    events['y'] = data['y']
    events['p'] = data['p'] * 2 - 1
    imgs = process(events)


print("DUR", time.time() - t0, np.shape(imgs))
       