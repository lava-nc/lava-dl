# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
from typing import Tuple, Union
import numpy as np
from PIL import Image
import glob


class PilotNetDataset():
    """Generic PilotNet dataset class. Returns image and ground truth value
    when the object is indexed.

    Parameters
    ----------
    path : str, optional
        Path of the dataset folder. If the folder does not exists, the folder
        is created and the dataset is downloaded and extracted to the folder.
        Defaults to '../data'.
    size : list, optional
        Size of the image. If it is not `200x66`, it is resized to the given
        value. Defaults to [200, 66].
    transform : lambda, optional
        Transformation function to be applied to the input image.
        Defaults to None.
    train : bool, optional
        Flag to indicate training or testing set. Defaults to True.
    visualize : bool, optional
        If true, the train/test split is ignored and the temporal sequence
        of the data is preserved. Defaults to False.
    sample_offset : int, optional
        sample offset. Default is 0.

    Usage
    -----

    >>> dataset = PilotNetDataset()
    >>> image, gt = dataeset[0]
    >>> num_samples = len(dataset)
    """
    def __init__(
        self,
        path: str = '../data',
        size: list = [200, 66],
        transform: Union[bool, None] = None,
        train: Union[bool, None] = True,
        visualize: Union[bool, None] = False,
        sample_offset: int = 0,
    ) -> None:
        self.path = os.path.join(path, 'driving_dataset')

        # check if dataset is available in path. If not download it
        if len(glob.glob(self.path)) == 0:  # dataset does not exist
            os.makedirs(path, exist_ok=True)

            print('Dataset not available locally. Starting download ...')
            id = '1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7'
            download_cmd = 'wget --load-cookies /tmp/cookies.txt '\
                + '"https://docs.google.com/uc?export=download&confirm='\
                + '$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate '\
                + f"'https://docs.google.com/uc?export=download&id={id}' -O- | "\
                + f"sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id={id}"\
                + f'" -O {path}/driving_dataset.zip && rm -rf /tmp/cookies.txt'
            print(download_cmd)
            os.system(download_cmd + f' >> {path}/download.log')
            print('Download complete.')
            print('Extracting data (this may take a while) ...')
            os.system(
                f'unzip {path}/driving_dataset.zip -d {path} >> '
                f'{path}/unzip.log'
            )
            print('Extraction complete.')

        with open(os.path.join(self.path, 'data.txt'), 'r') as data:
            all_samples = [line.split() for line in data]

        # this is what seems to be done in https://github.com/lhzlhz/PilotNet
        if visualize is True:
            inds = np.arange(len(all_samples))
            self.samples = [all_samples[i] for i in inds]
        else:
            inds = np.random.RandomState(seed=42).permutation(len(all_samples))
            if train is True:
                self.samples = [
                    all_samples[i] for i in inds[:int(len(all_samples) * 0.8)]
                ]
            else:
                self.samples = [
                    all_samples[i] for i in inds[-int(len(all_samples) * 0.2):]
                ]

        self.size = size
        self.transform = transform
        self.sample_offset = sample_offset

    def __getitem__(self, index: int) -> Tuple[np.ndarray, float]:
        index = (index + self.sample_offset) % len(self.samples)
        image = Image.open(
            os.path.join(self.path, self.samples[index][0])
        ).resize(self.size, resample=Image.BILINEAR)
        image = np.array(image) / 255
        if self.transform is not None:
            image = 2 * self.transform['weight'] * image \
                - self.transform['weight'] + self.transform['bias']
            image = image.astype(np.int32).transpose([1, 0, 2])
        ground_truth = float(self.samples[index][1])
        if ground_truth == 0:
            ground_truth = (
                float(self.samples[index - 1][1])
                + float(self.samples[index + 1][1])
            ) / 2
        gt_val = ground_truth * np.pi / 180
        return image.reshape(image.shape + (1,)), gt_val

    def __len__(self) -> int:
        return len(self.samples)
