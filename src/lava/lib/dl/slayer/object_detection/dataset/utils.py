# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple

import torch
from PIL.Image import Image, Transpose, Dict, Any

from ..boundingbox.utils import tensor_from_annotation

"""Dataset manipulation utility module."""


def flip_lr(image: Image) -> Image:
    """Flip a PIL image left-right.

    Parameters
    ----------
    image : Image
        Input image.

    Returns
    -------
    Image
        Flipped image.
    """
    return Image.transpose(image, Transpose.FLIP_LEFT_RIGHT)


def flip_ud(image: Image) -> Image:
    """Flip a PIL image up-down.

    Parameters
    ----------
    image : Image
        Input image.

    Returns
    -------
    Image
        Flipped image.
    """
    return Image.transpose(image, Transpose.FLIP_TOP_BOTTOM)


# TODO: This seems redundant: See if it breaks anything, otherwise remove.
# def collate_fn(
#         batch: List [Tuple[torch.tensor, Dict[Any, Any]]]
#     ) -> Tuple[torch.tensor, List[torch.tensor]]:
#     """Frames and annottation collate strategy for object detection.

#     Parameters
#     ----------
#     batch : List[Tuple[torch.tensor, Dict[Any, Any]]]
#         Raw batch collection of list of output from the dataset to be included
#         in a batch.

#     Returns
#     -------
#     Tuple[torch.tensor, List[torch.tensor]]
#         Stacked frames and annotation tensors ready to be used for object
#         detection.
#     """
#     images, targets = [], []

#     for image, annotation in batch:
#         images += [image]
#         targets += [[tensor_from_annotation(ann) for ann in annotation]]

#     T = len(targets[0])
#     targets = [[tgt[t] for tgt in targets] for t in range(T)]
#     return torch.stack(images), targets