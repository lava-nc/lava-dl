import os
import sys
import subprocess
import importlib
import random
from PIL import Image, ImageFilter, ImageTransform
from PIL.Image import Transpose

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from object_detection.dataset.utils import flip_lr, flip_ud
from object_detection.boundingbox import utils as bbutils
from object_detection.boundingbox.utils import Height, Width

from typing import Any, Dict, Tuple, Optional, Callable

try:
    from pycocotools.coco import COCO as COCOapi
except ModuleNotFoundError:
    if importlib.util.find_spec('cython') is None:
        subprocess.check_call([sys.executable,
                               '-m', 'pip', 'install', 'cython'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'git+https://github.com/philferriere/cocoapi.git'
                           '#egg=pycocotools&subdirectory=PythonAPI'])



def Image_Jitter(image, max_pixel_displacement_perc = 0.01):
    # max_pixel_displacement_perc = .01
    x,y = (torch.tensor(image.shape[1:3])*max_pixel_displacement_perc).type(torch.int)
    jitter_direction = random.randrange(-x,x), random.randrange(-y,y)    
    image_s = transforms.Pad(padding = (jitter_direction[0]*(jitter_direction[0]>0),
                        jitter_direction[1]*(jitter_direction[1]>0),
                        -jitter_direction[0]*(jitter_direction[0]<0),
                        -jitter_direction[1]*(jitter_direction[1]<0)))(image.squeeze())
    SS = image.size()[1:]
    ii = image_s[:, -jitter_direction[1]*(jitter_direction[1]<0):SS[0]-jitter_direction[1]*(jitter_direction[1]<0), 
    -jitter_direction[0]*(jitter_direction[0]<0):SS[1]-jitter_direction[0]*(jitter_direction[0]<0)]
    return image-ii.unsqueeze(-1)

from sklearn.cluster import MiniBatchKMeans
def quantize_global(image, k):
    k_means = MiniBatchKMeans(k, compute_labels=False)
    k_means.fit(image.reshape(-1, 1))
    labels = k_means.predict(image.reshape(-1, 1))
    q_img = k_means.cluster_centers_[labels]
    q_image = np.uint8(q_img.reshape(image.shape))
    return q_image

class _COCO(Dataset):
    def __init__(self,
                 root: str = '.',
                 train: bool = False) -> None:
        super().__init__()

        image_set = 'train' if train else 'val'
        self.coco = COCOapi(root + os.sep + 'annotations' + os.sep
                            + f'instances_{image_set}2017.json')
        self.root = root + os.sep + f'images{os.sep}{image_set}2017{os.sep}'
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_name = [d['name']
                         for d in self.coco.loadCats(self.coco.getCatIds())]
        self.super_cat_name = [d['supercategory']
                               for d in
                               self.coco.loadCats(self.coco.getCatIds())]
        self.idx_map = {name: idx for idx, name in enumerate(self.cat_name)}

    def __getitem__(self, index: int) -> Tuple[torch.tensor, Dict[Any, Any]]:
        id = self.ids[index]
        path = self.coco.loadImgs(id)[0]['file_name']
        image = Image.open(self.root + path).convert('RGB')
        width, height = image.size
        size = {'height': height, 'width': width}

        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        objects = []
        for ann in anns:
            name = self.coco.cats[ann['category_id']]['name']
            bndbox = {'xmin': ann['bbox'][0],
                      'ymin': ann['bbox'][1],
                      'xmax': ann['bbox'][0] + ann['bbox'][2],
                      'ymax': ann['bbox'][1] + ann['bbox'][3]}
            objects.append({'id': self.idx_map[name],
                            'name': name,
                            'bndbox': bndbox})

        annotation = {'size': size, 'object': objects}

        return image, {'annotation': annotation}

    def __len__(self) -> int:
        return len(self.ids)


class COCO(Dataset):
    def __init__(self,
                 root: str = './',
                 size: Tuple[Height, Width] = (448, 448),
                 train: bool = False,
                 augment_prob: float = 0.0,
                 image_jitter: bool = False) -> None:
        super().__init__()
        self.blur = transforms.GaussianBlur(kernel_size=5)
        self.color_jitter = transforms.ColorJitter()
        self.grayscale = transforms.Grayscale(num_output_channels=3)
        self.img_transform = transforms.Compose([transforms.Resize(size),
                                                 transforms.ToTensor()])
        self.bb_transform = transforms.Compose([
            lambda x: bbutils.resize_bounding_boxes(x, size),
        ])

        self.datasets = [_COCO(root=root, train=train)]
        self.classes = self.datasets[0].cat_name
        self.idx_map = self.datasets[0].idx_map
        self.augment_prob = augment_prob
        self.image_jitter = image_jitter

    def __getitem__(self, index) -> Tuple[torch.tensor, Dict[Any, Any]]:
        dataset_idx = index // len(self.datasets[0])
        index = index % len(self.datasets[0])
        image, annotation = self.datasets[dataset_idx][index]

        # flip left right
        if random.random() < self.augment_prob:
            image = Image.Image.transpose(image, Transpose.FLIP_LEFT_RIGHT)
            annotation = bbutils.fliplr_bounding_boxes(annotation)
        # # flip up down
        # if random.random() < self.augment_prob:
        #     image = Image.Image.transpose(image, Transpose.FLIP_TOP_BOTTOM)
        #     annotation = bbutils.flipud_bounding_boxes(annotation)
        # blur
        if random.random() < self.augment_prob:
            image = self.blur(image)
        # color jitter
        if random.random() < self.augment_prob:
            image = self.color_jitter(image)
        # grayscale
        if random.random() < self.augment_prob:
            image = self.grayscale(image)

        image = torch.unsqueeze(self.img_transform(image), -1)

        #jitter for mimicking DVS
        if self.image_jitter:
            image = Image_Jitter(image)
        
        annotation = self.bb_transform(annotation)

        return image, [annotation]  # list in time

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self.datasets])


   


