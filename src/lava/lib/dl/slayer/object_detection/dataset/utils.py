# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple

import torch
from PIL.Image import Image, Transpose
from torchvision import transforms
import random
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


def Image_Jitter(image: Image, max_pixel_displacement_perc=0.01) -> Image:
    """Subtracts a randmly jittered PIL image by itself.
    SLP: This method implement the Saccadic Like Perception (SLP) by mimicking
    on static images the response of the retinal ganglion cells that return 
    contours of the moving objects w.r.t. the observer. It is meant as a 
    training utility when intending to train on DVS like camera data. 

    Parameters
    ----------
    image : Image
        Input image.
    max_pixel_displacement_perc: float
        percentage of random jittering in the x and y image dimensions

    Returns
    -------
    Image
        Flipped image.
    """
    x, y = (torch.tensor(image.shape[1:3]) *
            max_pixel_displacement_perc).type(torch.int)
    while 1:
        jitter_direction = random.randrange(-x, x), random.randrange(-y, y)
        # ensure 0,0 displacement never happens
        if not (jitter_direction[0] == jitter_direction[1] == 0):
            break
    image_s = transforms.Pad(padding=(jitter_direction[0] * (jitter_direction[0] > 0),
                                        jitter_direction[1] *
                                        (jitter_direction[1] > 0),
                                        - jitter_direction[0] *
                                        (jitter_direction[0] < 0),
                                        - jitter_direction[1] *
                                        (jitter_direction[1] < 0)))(image.squeeze())
    SS = image.size()[1:]
    ii = image_s[:, 
                 -jitter_direction[1] * (jitter_direction[1] < 0):SS[0] 
                 - jitter_direction[1] * (jitter_direction[1] < 0),
                 - jitter_direction[0] * (jitter_direction[0] < 0):SS[1] 
                 - jitter_direction[0] * (jitter_direction[0] < 0)
                 ]
    return image - ii.unsqueeze(-1)
