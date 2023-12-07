import numpy as np
from lava.lib.dl.slayer import obd
import os
import multiprocessing
from joblib import Parallel, delayed


def single_data(dataset, idx, ds_path):
    name = dataset.get_name(idx) 
    images, annotations = dataset[idx]
    if not os.path.exists(ds_path + os.path.sep + name):
        os.makedirs(ds_path + os.path.sep + name)
        os.makedirs(ds_path + os.path.sep + name +  os.path.sep + 'events')
        os.makedirs(ds_path + os.path.sep + name +  os.path.sep + 'labels')
        
    idd = 0
    for events, label in zip(images, annotations):
        np.savez_compressed(ds_path + os.path.sep + name +  os.path.sep + 
                            'events' + os.path.sep + '{:05d}'.format(idd) + '.npz', a=events)
        np.savez_compressed(ds_path + os.path.sep + name +  os.path.sep + 
                                'labels' + os.path.sep + '{:05d}'.format(idd) + '.npz', a=label)
        idd += 1
    print(idx, '/', len(dataset), ' train_set: ', name)
    

if __name__ == '__main__':
    train_set = obd.dataset._PropheseeAutomotive(root='/home/lecampos/data/prophesee', 
                                                delta_t = 1,
                                                train=True, 
                                                randomize_seq= False,
                                                seq_len = 999999999999999)
    
    test_set = obd.dataset._PropheseeAutomotive(root='/home/lecampos/data/prophesee', 
                                                delta_t = 1,
                                                train=False, 
                                                randomize_seq= False,
                                                seq_len = 999999999999999)
                    
    out_path = '/home/lecampos/data/prophesee_small'
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_path = out_path + os.path.sep + '/train'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        
    test_path = out_path + os.path.sep + '/val'
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        
    num_cores = multiprocessing.cpu_count() - 1
    print('starting...')
    processed_list = Parallel(n_jobs=num_cores, prefer="threads")(delayed(single_data)(train_set, idx, train_path) for idx in range(len(train_set)))
    processed_list = Parallel(n_jobs=num_cores, prefer="threads")(delayed(single_data)(test_set, idx, test_path) for idx in range(len(test_set)))
        
    #for idx in range(len(train_set)):
    #    single_data(train_set, idx, train_path)
            
    #for idx in range(len(test_set)):
    #    single_data(test_set, idx, test_path)