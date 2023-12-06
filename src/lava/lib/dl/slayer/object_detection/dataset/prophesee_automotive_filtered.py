import os
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import resize_events_frame, fliplr_events
from ..boundingbox import utils as bbutils
from ..boundingbox.utils import Height, Width

from typing import Any, Dict, Tuple

class _PropheseeAutomotiveFiltered(Dataset):
    def __init__(self,
                 root: str = '.',
                 seq_len: int = 32,
                 randomize_seq: bool = False,
                 train: bool = False) -> None:
        super().__init__()
        
        self.cat_name = []
        self.seq_len = seq_len
        self.randomize_seq = randomize_seq
        
        with open(root + os.sep + 'label_map_dictionary.json') as file:
            data = json.load(file)
            self.idx_map = {int(key) : value for key, value in data.items()}
            [self.cat_name.append(value) for _, value in data.items()]

        dataset = 'train' if train else 'val'
        self.dataset_path = root + os.sep + dataset
        tmp_files = os.listdir(self.dataset_path)
        
        
        # validate frames size
        self.files = []
        for file in tmp_files:
            videos_path = self.dataset_path + os.path.sep + file + os.path.sep + 'events'
            bbox_path = self.dataset_path + os.path.sep + file + os.path.sep + 'labels'
            
            if len( os.listdir(videos_path)) >= self.seq_len and len(os.listdir(bbox_path)) >= self.seq_len:
                self.files.append(file)
        
        self.files.sort()


    def __getitem__(self, index: int) -> Tuple[torch.tensor, Dict[Any, Any]]:
        
        file_name = self.files[index]
        
        videos_path = self.dataset_path + os.path.sep + file_name + os.path.sep + 'events'
        bbox_path = self.dataset_path + os.path.sep + file_name + os.path.sep + 'labels'
        
        videos_list = os.listdir(videos_path)
        videos_list.sort()
        bbox_list = os.listdir(bbox_path)
        bbox_list.sort()
        
        #if len(videos_list) < self.seq_len or len(bbox_path) < self.seq_len:
        #    return [], []
        
        id_load = 0
        if self.randomize_seq:
            if len(videos_list) > self.seq_len:
                skip_time = len(videos_list) - (self.seq_len + 1)
                id_load = np.random.randint(0, skip_time)
        
        images = []
        annotations = []
        for idx in range(id_load, len(videos_list)):
            images.append(np.load(videos_path + os.path.sep + videos_list[idx])['a'])
            annotations.append(np.load(bbox_path + os.path.sep + bbox_list[idx], allow_pickle='TRUE')['a'].item())
  
        return images, annotations

    def __len__(self) -> int:
        return len(self.files)


class PropheseeAutomotiveFiltered(Dataset):
    def __init__(self,
                 root: str = './',
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 seq_len: int = 32,
                 randomize_seq: bool = False,
                 augment_prob: float = 0.0) -> None:
        super().__init__()
        self.img_transform = transforms.Compose([lambda x: resize_events_frame(x, size),
                                                 lambda x: torch.FloatTensor(x).permute([2, 0, 1])])
        self.bb_transform = transforms.Compose([
            lambda x: bbutils.resize_bounding_boxes(x, size),
        ])

        self.datasets = [_PropheseeAutomotiveFiltered(root=root, train=train,
                                              seq_len=seq_len, randomize_seq=randomize_seq)]

        self.classes = self.datasets[0].cat_name
        self.idx_map = self.datasets[0].idx_map
        self.augment_prob = augment_prob
        self.seq_len = seq_len
        self.randomize_seq = randomize_seq

    def __getitem__(self, index) -> Tuple[torch.tensor, Dict[Any, Any]]:
        
        dataset_idx = index // len(self.datasets[0])
        index = index % len(self.datasets[0])
        images, annotations = self.datasets[dataset_idx][index]
        
        
        # while True:
        #     images, annotations = self.datasets[dataset_idx][index]
        #     if (len(images) == self.seq_len) and (len(annotations) == self.seq_len):
        #         break
        #     index = random.randint(0, (len(self.datasets[0]) - 1) )

        # flip left right
        if random.random() < self.augment_prob:
            for idx in range(len(images)):
                images[idx] = fliplr_events(images[idx])
                annotations[idx] = bbutils.fliplr_bounding_boxes(
                    annotations[idx])
        
        image = torch.cat([torch.unsqueeze(self.img_transform(img), -1)
                           for img in images], dim=-1)
        annotations = [self.bb_transform(ann) for ann in annotations]

        # [C, H, W, T], [bbox] * T
        # list in time
        return image, annotations

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])
