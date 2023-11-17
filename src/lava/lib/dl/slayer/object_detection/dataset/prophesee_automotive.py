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

try:
    from src.io.psee_loader import PSEELoader
except ModuleNotFoundError:
    print(" Error! ")



class _PropheseeAutomotive(Dataset):
    def __init__(self,
                 root: str = '.',
                 delta_t: int = 1,
                 skip: int = 0,
                 train: bool = False) -> None:
        super().__init__()
        
        self.cat_name = []
        self.delta_t = delta_t * 1000
        self.skip = skip
        
        with open(root + os.sep + 'label_map_dictionary.json') as file:
            data = json.load(file)
            self.idx_map = {int(key) : value for key, value in data.items()}
            [self.cat_name.append(value) for _, value in data.items()]

        dataset = 'train' if train else 'val'
        self.dataset_path = root + os.sep + dataset
        
        td_files = [td_file for td_file in os.listdir(self.dataset_path) if td_file.endswith('.dat')]
        self.videos = [PSEELoader(self.dataset_path + os.sep + td_file) for td_file in td_files]
        self.bbox_videos = [PSEELoader(self.dataset_path + os.sep + td_file.split('_td.dat')[0] +  '_bbox.npy') for td_file in td_files]    

    def __getitem__(self, index: int) -> Tuple[torch.tensor, Dict[Any, Any]]:
        video = self.videos[index]
        bbox_video = self.bbox_videos[index]
        height, width = video.get_size()
        images = []
        annotations = []

        while not video.done:
            events = video.load_delta_t(self.delta_t)
            frame = np.zeros((height, width, 2), dtype=np.uint8)
            if events['x'].max() > width:
                print( events['x'].min())
            if events['y'].max() > height:
                print( events['y'].min())
            
            valid = np.all((events['x'] >= 0 ) & (events['x'] < width) & (events['y'] >= 0 ) & (events['y'] < height))
            events = events[valid]
            
            if len(events) < 1:
                print(len(events))
            
            #drop if out of FoV
            if events['x'].max() > width:
                continue
            if events['y'].max() > height:
                continue
            
            frame[events['y'][events['p'] == 1], events['x'][events['p'] == 1], 0] = 1
            frame[events['y'][events['p'] == 0], events['x'][events['p'] == 0], 1] = 1
           
            boxes = bbox_video.load_delta_t(self.delta_t)
            objects = []
            size = {'height': height, 'width': width}
            
            for idx in range(boxes.shape[0]):
                if (int(boxes['w'][idx]) > 0) and (int(boxes['h'][idx]) > 0):
                    bndbox = {'xmin': int(boxes['x'][idx]),
                                'ymin': int(boxes['y'][idx]),
                                'xmax': int(boxes['x'][idx]) + int(boxes['w'][idx]),
                                'ymax': int(boxes['y'][idx]) + int(boxes['h'][idx])}
                    name = self.idx_map[boxes['class_id'][idx]]
                    if (bndbox['xmax'] < width) and (bndbox['ymax']  < height) and (bndbox['xmin'] > 0) and (bndbox['ymin'] > 0):
                        objects.append({'id': boxes['class_id'][idx],
                                        'name': name,
                                        'bndbox': bndbox})
            if len(objects) > 0: 
                annotation = {'size': size, 'object': objects}
                images.append(frame)
                annotations.append({'annotation': annotation})
                
        if len(images) < 1:
            print(self.videos[index])
            
        return images, annotations

    def __len__(self) -> int:
        return len(self.videos)


class PropheseeAutomotive(Dataset):
    def __init__(self,
                 root: str = './',
                 delta_t: int = 1, 
                 skip: int = 0,
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 seq_len: int = 32,
                 randomize_seq: bool = False,
                 augment_prob: float = 0.0) -> None:
        super().__init__()
        self.img_transform = transforms.Compose([lambda x: resize_events_frame(x, size),
                                                 transforms.ToTensor()])
        self.bb_transform = transforms.Compose([
            lambda x: bbutils.resize_bounding_boxes(x, size),
        ])

        self.datasets = [_PropheseeAutomotive(root=root, delta_t=delta_t, skip=skip, train=train)]

        self.classes = self.datasets[0].cat_name
        self.idx_map = self.datasets[0].idx_map
        self.augment_prob = augment_prob
        self.seq_len = seq_len
        self.randomize_seq = randomize_seq

    def __getitem__(self, index) -> Tuple[torch.tensor, Dict[Any, Any]]:
        
        dataset_idx = index // len(self.datasets[0])
        index = index % len(self.datasets[0])
        images, annotations = self.datasets[dataset_idx][index]

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
        num_seq = image.shape[-1]
        
        if self.randomize_seq:
            id_rand = (num_seq - self.seq_len) if num_seq > self.seq_len else num_seq
            start_idx = np.random.randint(id_rand)
        else:
            start_idx = 0
        stop_idx = start_idx + self.seq_len
        stop_idx = stop_idx if stop_idx < image.shape[-1] else image.shape[-1]

        # list in time
        return image[..., start_idx:stop_idx], annotations[start_idx:stop_idx]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])


if __name__ == '__main__':
    train_set = PropheseeAutomotive(root='/home/lecampos/lava-dev/data/prophesee', train=True)
    print(f'{len(train_set) = }')
    print(f'{train_set.classes = }')
    print(f'{train_set.idx_map = }')
    
    image, annotation = train_set[0]

    for _ in range(5):
        image, annotation = train_set[40]
        # image, annotation = train_set[np.random.randint(len(train_set))]
        # display(mark_bounding_boxes(image,
        #                            annotation[0]['annotation']['object'],
        #                            thickness=1))
