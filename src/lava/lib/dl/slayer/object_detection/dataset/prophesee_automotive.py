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
    from src.io.box_filtering import filter_boxes
except ModuleNotFoundError:
    print(" Error! ")


class _PropheseeAutomotive(Dataset):
    def __init__(self,
                 root: str = '.',
                 delta_t: int = 1,
                 seq_len: int = 32,
                 events_ratio: float = 0.07,
                 randomize_seq: bool = False,
                 train: bool = False) -> None:
        super().__init__()
        self.cat_name = []
        self.delta_t = delta_t * 1000
        self.seq_len = seq_len
        self.randomize_seq = randomize_seq
        self.events_ratio_threshold = events_ratio

        with open(root + os.sep + 'label_map_dictionary.json') as file:
            data = json.load(file)
            self.idx_map = {int(key) : value for key, value in data.items()}
            [self.cat_name.append(value) for _, value in data.items()]

        dataset = 'train' if train else 'val'
        self.dataset_path = root + os.sep + dataset

        td_files = [td_file
                    for td_file in os.listdir(self.dataset_path)
                    if td_file.endswith('.dat')]
        self.videos = [PSEELoader(self.dataset_path + os.sep + td_file)
                       for td_file in td_files]
        self.bbox_videos = [
            PSEELoader(self.dataset_path + os.sep
                       + td_file.split('_td.dat')[0]
                       + '_bbox.npy') for td_file in td_files]    

    def validate_bbox(self, events, bbox):
        events_bbox = events[bbox['ymin']:bbox['ymax'],
                             bbox['xmin']:bbox['xmax']]
        pixels_area = (bbox['xmax'] - bbox['xmin']) * \
            (bbox['ymax'] - bbox['ymin'])
        events_ratio = np.count_nonzero(events_bbox) / pixels_area
        return events_ratio > self.events_ratio_threshold

    def get_seq(self, video, bbox_video):
        images = []
        annotations = []
        height, width = video.get_size()

        while not video.done:
            try:
                events = video.load_delta_t(self.delta_t)
                boxes = bbox_video.load_delta_t(self.delta_t)
            except (AssertionError, IndexError):
                pass

            min_box_diag = 60
            min_box_side = 20
            boxes = filter_boxes(boxes, int(1e5), min_box_diag, min_box_side)

            frame = np.zeros((height, width, 2), dtype=np.uint8)
            valid = (events['x'] >= 0 ) & (events['x'] < width) & \
                    (events['y'] >= 0 ) & (events['y'] < height)
            events = events[valid]
            frame[events['y'][events['p'] == 1],
                  events['x'][events['p'] == 1], 0] = 1
            frame[events['y'][events['p'] == 0],
                  events['x'][events['p'] == 0], 1] = 1

            objects = []
            size = {'height': height, 'width': width}

            for idx in range(boxes.shape[0]):
                if (int(boxes['w'][idx]) > 0) and (int(boxes['h'][idx]) > 0):
                    bndbox = {
                                'xmin': int(boxes['x'][idx]),
                                'ymin': int(boxes['y'][idx]),
                                'xmax': int(boxes['x'][idx])
                                + int(boxes['w'][idx]),
                                'ymax': int(boxes['y'][idx])
                                + int(boxes['h'][idx])}
                    name = self.idx_map[boxes['class_id'][idx]]
                    if (bndbox['xmax'] < width) and \
                        (bndbox['ymax'] < height) and \
                            (bndbox['xmin'] > 0) and (bndbox['ymin'] > 0):
                        if len(images) == 0:
                            if self.validate_bbox(frame, bndbox):
                                objects.append({'id': boxes['class_id'][idx],
                                                'name': name,
                                                'bndbox': bndbox})
                        else:
                            objects.append({'id': boxes['class_id'][idx],
                                                'name': name,
                                                'bndbox': bndbox})

            if len(objects) == 0:
                if len(annotations) == 0:
                    continue
                annotations.append(annotations[-1])
            else:
                annotation = {'size': size, 'object': objects}
                annotations.append({'annotation': annotation})

            images.append(frame)

            if len(images) >= self.seq_len:
                break
        return images, annotations

    def get_name(self, index):
        video = self.videos[index]
        return video._file.name.split('_td.dat')[0].split('/')[-1]

    def __getitem__(self, index: int) -> Tuple[torch.tensor, Dict[Any, Any]]:
        video = self.videos[index]
        bbox_video = self.bbox_videos[index]

        if self.randomize_seq:
            skip_time = (video.duration_s - 0.1) - \
                ((self.seq_len * self.delta_t) / 1000000)
            while True:
                try:
                    video.seek_time(skip_time * np.random.random() * 1000000)
                    bbox_video.seek_time(skip_time
                                         * np.random.random() * 1000000)
                    break
                except IndexError:
                    continue

        images, annotations = self.get_seq(video, bbox_video)

        if len(images) != self.seq_len or len(annotations) != self.seq_len:
            video.reset()
            bbox_video.reset()
            images, annotations = self.get_seq(video, bbox_video)
        return images, annotations

    def __len__(self) -> int:
        return len(self.videos)


class PropheseeAutomotive(Dataset):
    def __init__(self,
                 root: str = './',
                 delta_t: int = 1,
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 seq_len: int = 32,
                 events_ratio: float = 0.07,
                 randomize_seq: bool = False,
                 augment_prob: float = 0.0) -> None:
        super().__init__()
        self.img_transform = transforms.Compose([
            lambda x: resize_events_frame(x, size),
            lambda x: torch.FloatTensor(x).permute([2, 0, 1])])
        self.bb_transform = transforms.Compose([
            lambda x: bbutils.resize_bounding_boxes(x, size),
        ])

        self.datasets = [_PropheseeAutomotive(root=root,
                                              delta_t=delta_t,
                                              train=train,
                                              events_ratio=events_ratio,
                                              seq_len=seq_len,
                                              randomize_seq=randomize_seq)]

        self.classes = self.datasets[0].cat_name
        self.idx_map = self.datasets[0].idx_map
        self.augment_prob = augment_prob
        self.seq_len = seq_len

    def __getitem__(self, index) -> Tuple[torch.tensor, Dict[Any, Any]]:

        dataset_idx = index // len(self.datasets[0])
        index = index % len(self.datasets[0])
        images, annotations = [], []
        while (len(images) != self.seq_len) and \
                (len(annotations) != self.seq_len):
            images, annotations = self.datasets[dataset_idx][index]
            index = random.randint(0, len(self.datasets[0]) - 1)

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
