# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import torch
from PIL.Image import Image, Transpose
from ..boundingbox.utils import tensor_from_annotation


def flip_lr(image: Image) -> Image:
    return Image.transpose(image, Transpose.FLIP_LEFT_RIGHT)


def flip_ud(image: Image) -> Image:
    return Image.transpose(image, Transpose.FLIP_TOP_BOTTOM)


def collate_fn(batch):
    images, targets = [], []

    for image, annotation in batch:
        images += [image]
        targets += [[tensor_from_annotation(ann) for ann in annotation]]

    T = len(targets[0])
    targets = [[tgt[t] for tgt in targets] for t in range(T)]
    return torch.stack(images), targets