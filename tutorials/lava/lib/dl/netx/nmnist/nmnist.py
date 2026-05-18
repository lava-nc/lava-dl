# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause
import glob
import os
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

import lava.lib.dl.slayer as slayer


def augment(event):
    x_shift = 4
    y_shift = 4
    theta = 10
    xjitter = np.random.randint(2*x_shift) - x_shift
    yjitter = np.random.randint(2*y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event


class NMNISTDataset():
    """NMNIST dataset method

    Parameters
    ----------
    path : str, optional
        path of dataset root, by default 'data'
    train : bool, optional
        train/test flag, by default True
    sampling_time : int, optional
        sampling time of event data, by default 1
    sample_length : int, optional
        length of sample data, by default 300
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default Noney.
    download : bool, optional
        enable/disable automatic download, by default True
    """
    def __init__(
        self, path='data',
        train=True,
        sampling_time=1, sample_length=300,
        transform=None, download=True,
    ):
        super(NMNISTDataset, self).__init__()
        self.path = path
        
        if train:
            data_path = path + '/Train'
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/'\
                'AABlMOuR15ugeOxMCX0Pvoxga/Train.zip'
        else:
            data_path = path + '/Test'
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/'\
                'AADSKgJ2CjaBWh75HnTNZyhca/Test.zip'

        if download is True:
            attribution_text = '''
            NMNIST dataset is freely available here:
            https://www.garrickorchard.com/datasets/n-mnist

            (c) Creative Commons:
            Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
            "Converting Static Image Datasets to Spiking Neuromorphic Datasets Using
            Saccades",
            Frontiers in Neuroscience, vol.9, no.437, Oct. 2015
            '''.replace(' '*12, '')
            if train is True:
                print(attribution_text)

            if len(glob.glob(f'{data_path}/')) == 0:  # dataset does not exist
                print(
                    f'NMNIST {"training" if train is True else "testing"} '
                    'dataset is not available locally.'
                )
                print('Attempting download (This will take a while) ...')
                os.system(f'wget {source} -P {self.path}/ -q --show-progress')
                print('Extracting files ...')
                with zipfile.ZipFile(data_path + '.zip') as zip_file:
                    for member in zip_file.namelist():
                        zip_file.extract(member, self.path)
                print('Download complete.')
        else:
            assert len(glob.glob(f'{data_path}/')) == 0, \
                f'Dataset does not exist. Either set download=True '\
                f'or download it from '\
                f'https://www.garrickorchard.com/datasets/n-mnist '\
                f'to {data_path}/'

        self.samples = glob.glob(f'{data_path}/*/*.bin')  #TODO
     
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length/sampling_time)
        self.transform = transform
       
    def __getitem__(self, i):
        filename = self.samples[i]
        label = int(filename.split('/')[-2])
        event = slayer.io.read_2d_spikes(filename)
        if self.transform is not None:
            event = self.transform(event)
        spike = event.fill_tensor(
                np.zeros((2, 34, 34, self.num_time_bins)),
                sampling_time=self.sampling_time,
            )
        return spike.reshape(-1, self.num_time_bins), label

    def __len__(self):
        return len(self.samples)

