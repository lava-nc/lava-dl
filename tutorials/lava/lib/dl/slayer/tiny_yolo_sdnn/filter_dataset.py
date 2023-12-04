import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import IPython.display as ipd
from lava.lib.dl.slayer import obd
import os

if __name__ == '__main__':
    train_set = obd.dataset._PropheseeAutomotive(root='/data-raid/sshresth/data/Prophesee_1mp', 
                                                delta_t = 1,
                                                train=True, 
                                                randomize_seq= False,
                                                seq_len = 999999999)
    
    test_set = obd.dataset._PropheseeAutomotive(root='/data-raid/sshresth/data/Prophesee_1mp', 
                                                delta_t = 1,
                                                train=False, 
                                                randomize_seq= False,
                                                seq_len = 999999999)
                    

    out_path = '/data-raid/sshresth/data/Prophesee_fl'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_path = '/data-raid/sshresth/data/Prophesee_fl' + '/train'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        
    test_path = '/data-raid/sshresth/data/Prophesee_fl' + '/val'
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        
    for idx in range(len(train_set)):
        name = train_set.get_name(idx) 
        images, annotations = train_set[idx]
        if not os.path.exists(train_path + os.path.sep + name):
            os.makedirs(train_path + os.path.sep + name)
            os.makedirs(train_path + os.path.sep + name +  os.path.sep + 'events')
            os.makedirs(train_path + os.path.sep + name +  os.path.sep + 'labels')
        idx = 0
        for events, label in zip(images, annotations):
            np.savez_compressed(train_path + os.path.sep + name +  os.path.sep + 
                                'events' + os.path.sep + '{:05d}'.format(idx) + '.npz', a=events)
            np.savez_compressed(train_path + os.path.sep + name +  os.path.sep + 
                                    'labels' + os.path.sep + '{:05d}'.format(idx) + '.npz', a=label)
            
            events_loaded = np.load(train_path + os.path.sep + name +  os.path.sep + 
                                'events' + os.path.sep + '{:05d}'.format(idx) + '.npz')['a']

            label_loaded = np.load(train_path + os.path.sep + name +  os.path.sep + 
                                'labels' + os.path.sep + '{:05d}'.format(idx) + '.npz',
                                allow_pickle='TRUE')['a'].item()
            idx += 1
        print('train_set: ', name)
            
    for idx in range(len(test_set)):
        name = test_set.get_name(idx) 
        images, annotations = test_set[idx]
        if not os.path.exists(test_path + os.path.sep + name):
            os.makedirs(test_path + os.path.sep + name)
            os.makedirs(test_path + os.path.sep + name +  os.path.sep + 'events')
            os.makedirs(test_path + os.path.sep + name +  os.path.sep + 'labels')
        idx = 0
        for events, label in zip(images, annotations):
            np.savez_compressed(test_path + os.path.sep + name +  os.path.sep + 
                                'events' + os.path.sep + '{:05d}'.format(idx) + '.npz', a=events)
            np.savez_compressed(test_path + os.path.sep + name +  os.path.sep + 
                                    'labels' + os.path.sep + '{:05d}'.format(idx) + '.npz', a=label)
            
            events_loaded = np.load(test_path + os.path.sep + name +  os.path.sep + 
                                'events' + os.path.sep + '{:05d}'.format(idx) + '.npz')['a']

            label_loaded = np.load(test_path + os.path.sep + name +  os.path.sep + 
                                'labels' + os.path.sep + '{:05d}'.format(idx) + '.npz',
                                allow_pickle='TRUE')['a'].item()
            idx += 1
        print('test_set: ', name)