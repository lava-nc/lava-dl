# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import Callable, List, Tuple, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .boundingbox.metrics import bbox_ciou, bbox_iou, wh_iou
from .boundingbox.utils import tensor_from_annotation

"""Base YOLO methods and classes definiton."""


def _yolo(x: torch.tensor,
          anchors: torch.tensor,
          clamp_max: float = 5.0) -> torch.tensor:
    # converts raw predictions to bounding box predictions.
    _, _, H, W, _, _ = x.shape
    range_y, range_x = torch.meshgrid(
        torch.arange(H, dtype=x.dtype, device=x.device),
        torch.arange(W, dtype=x.dtype, device=x.device),
        indexing='ij',
    )
    anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

    x_center = (torch.sigmoid(x[:, :, :, :, 0:1, :])
                + range_x[None, None, :, :, None, None]) / W
    y_center = (torch.sigmoid(x[:, :, :, :, 1:2, :])
                + range_y[None, None, :, :, None, None]) / H
    width = (torch.exp(
        x[:, :, :, :, 2:3, :].clamp(
            max=clamp_max)) * anchor_x[None, :, None, None, None, None]) / W
    height = (torch.exp(
        x[:, :, :, :, 3:4, :].clamp(
            max=clamp_max)) * anchor_y[None, :, None, None, None, None]) / H
    confidence = torch.sigmoid(x[:, :, :, :, 4:5, :])
    classes = torch.softmax(x[:, :, :, :, 5:, :], dim=-2)

    x = torch.concat([x_center, y_center, width, height,
                      confidence, classes], dim=-2)

    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f'{torch.isnan(x_center).any()=}')
        print(f'{torch.isinf(x_center).any()=}')
        print(f'{torch.isnan(y_center).any()=}')
        print(f'{torch.isinf(y_center).any()=}')
        print(f'{torch.isnan(width).any()=}')
        print(f'{torch.isinf(width).any()=}')
        print(f'{torch.isnan(height).any()=}')
        print(f'{torch.isinf(height).any()=}')
        raise RuntimeError('Ecountered NaN and Inf!')

    return x  # batch, anchor, height, width, predictions, time


def _yolo_target(x: torch.tensor, anchors: torch.tensor) -> torch.tensor:
    # Generates target tensor for a given anchor resolution.
    _, _, H, W, _ = x.shape
    range_y, range_x = torch.meshgrid(
        torch.arange(H, dtype=x.dtype, device=x.device),
        torch.arange(W, dtype=x.dtype, device=x.device),
        indexing='ij',
    )
    anchors = torch.ones_like(anchors)
    anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

    x_center = ((x[:, :, :, :, 0:1])
                + range_x[None, None, :, :, None]) / W
    y_center = ((x[:, :, :, :, 1:2])
                + range_y[None, None, :, :, None]) / H
    width = ((x[:, :, :, :, 2:3]) * anchor_x[None, :, None, None, None]) / W
    height = ((x[:, :, :, :, 3:4]) * anchor_y[None, :, None, None, None]) / H
    confidence = (x[:, :, :, :, 4:5])
    labels = x[:, :, :, :, 5]
    classes = F.one_hot(labels.long(), num_classes=x.shape[-1] - 5)

    x = torch.concat([x_center, y_center, width, height,
                      confidence, classes], dim=-1).unsqueeze(dim=-1)

    return x  # batch, anchor, height, width, predictions, time


class YOLOtarget:
    """YOLO target generation module.

    Parameters
    ----------
    anchors : Tuple[Tuple]
        Anchor points.
    scales : Tuple[Tuple[int]]
        Prediction scales.
    num_classes : int
        Number of object classes.
    ignore_iou_thres : float, optional
        IOU overlap threshold to ignore target, by default 0.5.
    """
    def __init__(self,
                 anchors: Tuple[Tuple],
                 scales: Tuple[Tuple[int]],  # x, y format
                 num_classes: int,
                 ignore_iou_thres: float = 0.5) -> None:
        if len(anchors) != len(scales):
            raise RuntimeError(f'Number of anchors and number of scales '
                               f'do not match. Found {anchors=} and'
                               f'{scales=}')
        if not torch.is_tensor(anchors):
            anchors = torch.FloatTensor(anchors)
        else:
            anchors = anchors.cpu()
        self.anchors = anchors
        self.scales = scales
        self.num_classes = num_classes
        self.flat_anchors = torch.concat([a for a in anchors])
        self.ignore_iou_thres = ignore_iou_thres
        self.num_scales, self.num_anchors, *_ = anchors.shape

    def forward(self, targets: torch.tensor) -> torch.tensor:
        """Translates bounding box tensor to YOLO target tensor.

        Parameters
        ----------
        targets : torch.tensor
            Bounding box targets.

        Returns
        -------
        torch.tensor
            Expanded YOLO targets.
        """
        tgts = []
        for (W, H) in self.scales:
            tgts.append(torch.zeros(self.num_anchors, H, W, 6))

        for obj in range(targets.shape[0]):
            tgt = targets[obj: obj + 1]
            iou_anchors = wh_iou(tgt[..., 2:4], self.flat_anchors).flatten()
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, conf, label = tgt[0]
            has_anchor = [False] * len(self.scales)

            for anchor_idx in anchor_indices.tolist():
                scale_idx = anchor_idx // self.anchors.shape[1]
                anchor_on_scale = anchor_idx % self.anchors.shape[1]
                scale_x, scale_y = self.scales[scale_idx]
                i, j = int(y * scale_y), int(x * scale_x)
                anchor_taken = tgts[scale_idx][anchor_on_scale, i, j, 4]

                if not anchor_taken and not has_anchor[scale_idx]:
                    tgts[scale_idx][anchor_on_scale, i, j, 4] = 1
                    y_cell = scale_y * y - i
                    x_cell = scale_x * x - j
                    width_cell = width * scale_x
                    height_cell = height * scale_y
                    box_coordinates = torch.tensor([x_cell,
                                                    y_cell,
                                                    width_cell,
                                                    height_cell])
                    tgts[scale_idx][anchor_on_scale, i, j, :4] = box_coordinates
                    tgts[scale_idx][anchor_on_scale, i, j, 5] = int(label)
                    has_anchor[scale_idx] = True
                elif not (anchor_taken
                          and iou_anchors[anchor_idx] > self.ignore_iou_thres):
                    tgts[scale_idx][anchor_on_scale, i, j, 4] = -1

        return tgts

    def collate_fn(self,
                   batch: List[Tuple[torch.tensor, Dict[Any, Any]]]
                   ) -> Tuple[torch.tensor,
                              List[torch.tensor], List[List[torch.tensor]]]:
        """Frames and annottation collate strategy for object detection.

        Parameters
        ----------
        batch : List[Tuple[torch.tensor, Dict[Any, Any]]]
            Raw batch collection of list of output from the dataset to be
            included in a batch.

        Returns
        -------
        Tuple[torch.tensor, List[torch.tensor]]
            Stacked frames, target tensor, and annotation tensors for
            YOLO object detection.
        """
        images, targets, bboxes = [], [], []

        for image, annotation in batch:
            images += [image]
            bbox = [tensor_from_annotation(ann) for ann in annotation]
            tgt = [self.forward(b) for b in bbox]
            bboxes += [bbox]
            targets += [[torch.stack([tgt[time][scale]
                                      for time in range(len(tgt))], dim=-1)
                         for scale in range(self.num_scales)]]

        T = len(bboxes[0])
        bboxes = [[bbox[t] for bbox in bboxes] for t in range(T)]

        return (torch.stack(images),
                [torch.stack([targets[batch][scale]
                              for batch in range(len(targets))])
                 for scale in range(self.num_scales)], bboxes)


class YOLOLoss(torch.nn.Module):
    """YOLO Loss module.

    Parameters
    ----------
    anchors : Tuple[Tuple]
        Prediction achor coordinates.
    lambda_coord : float, optional
        Coordinate loss factor, by default 1.
    lambda_noobj : float, optional
        No object loss factor, by default 10.0.
    lambda_obj : float, optional
        Object loss factor, by default 5.0.
    lambda_cls : float, optional
        Class loss factor, by default 1.0.
    lambda_iou : float, optional
        IoU loss factor, by default 1.0.
    alpha_iou : float, optional
        IoU mixture ratio, by default 0.25.
    startup_samples : int, optional
        Number of iterations to consider as startup, by default 10000.
    label_smoothing : float, optional
        Label smoothing factor, by default 0.1.
    """
    def __init__(self,
                 anchors: Tuple[Tuple],
                 lambda_coord: float = 1,
                 lambda_noobj: float = 10.0,
                 lambda_obj: float = 5.0,
                 lambda_cls: float = 1.0,
                 lambda_iou: float = 1.0,
                 alpha_iou: float = 0.25,
                 startup_samples: int = 10000,
                 label_smoothing: float = 0.1,
                 ) -> None:
        super().__init__()

        if not torch.is_tensor(anchors):
            anchors = torch.FloatTensor(anchors)
        self.register_buffer('anchors', anchors)
        self.num_scales, self.num_anchors, *_ = anchors.shape
        self.flat_anchors = torch.concat([a for a in anchors])

        self.mse = torch.nn.MSELoss(reduction='mean')
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.cel = torch.nn.CrossEntropyLoss(reduction='mean',
                                             label_smoothing=label_smoothing)

        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)
        self.lambda_obj = float(lambda_obj)
        self.lambda_cls = float(lambda_cls)
        self.lambda_iou = float(lambda_iou)
        self.alpha_iou = float(alpha_iou)

        self.startup_samples = startup_samples
        self.samples = 0

    def forward(self,
                predictions: List[torch.tensor],
                targets: List[torch.tensor]) -> torch.tensor:
        """Loss evaluation method.

        Parameters
        ----------
        predictions : List[torch.tensor]
            Prediction tensor at different scales for each head.
        targets : List[torch.tensor]
            Target tensor at different sclaes for each head.

        Returns
        -------
        torch.tensor
            Loss value.
        """
        loss = 0
        loss_distr = []

        for time in range(predictions[0].shape[-1]):
            prediction = [p[..., time].clone() for p in predictions]
            target = [t[..., time] for t in targets]
            for p, t, a in zip(prediction, target, self.anchors):
                l, ld = self.forward_scale(p, t.to(p.device), a.to(p.device))
                loss += l
                loss_distr.append(ld)

        self.samples += predictions[0].shape[0]
        return loss, torch.tensor(loss_distr).sum(dim=0)

    def forward_scale(self,
                      predictions: torch.tensor,
                      targets: torch.tensor,
                      anchors: torch.tensor) -> torch.tensor:
        """Loss evaluation for a prediction head of YOLO.

        Parameters
        ----------
        predictions : torch.tensor
            Prediction at a selected scale.
        targets : torch.tensor
            Target at a selected scale.
        anchors : torch.tensor
            Anchors at a selected scale.

        Returns
        -------
        torch.tensor
            Loss for a prediction scale.
        """
        obj = targets[..., 4] == 1
        noobj = targets[..., 4] == 0

        # No object loss
        # bce converges better
        no_object_loss = self.bce(predictions[..., 4:5][noobj],
                                  targets[..., 4:5][noobj])

        # Object loss
        anchors = anchors.reshape(1, -1, 1, 1, 2)
        box_preds = torch.cat([torch.sigmoid(predictions[..., 0:2]),
                               torch.exp(predictions[..., 2:4]) * anchors],
                              dim=-1)
        ious = bbox_ciou(box_preds[..., :4][obj],
                         targets[..., :4][obj])

        alpha = self.alpha_iou
        scale = (1 - alpha) + alpha * ious.detach().clamp(0)

        # bce converges better
        object_loss = self.bce(predictions[..., 4:5][obj].flatten(),
                               scale * targets[..., 4:5][obj].flatten())

        # IOU loss
        # mean squared error converges better
        iou_loss = torch.mean((1 - ious)**2)

        # Coordinate loss
        predictions[..., 0:2] = predictions[..., 0:2].sigmoid()
        targets[..., 2:4] = torch.log(targets[..., 2:4] / anchors + 1e-16)
        coord_loss = self.mse(predictions[..., :4][obj], targets[..., :4][obj])

        # Class loss
        cls_loss = self.cel(predictions[..., 5:][obj],
                            targets[..., 5][obj].long())

        loss_distr = [self.lambda_coord * coord_loss,
                      self.lambda_obj * object_loss,
                      self.lambda_noobj * no_object_loss,
                      self.lambda_cls * cls_loss,
                      self.lambda_iou * iou_loss]

        return sum(loss_distr), loss_distr


class YOLOBase(torch.nn.Module):
    """Base YOLO network class for temporal processing.

    Parameters
    ----------
    num_classes : int, optional
        Number of object classes to predict, by default 20.
    anchors : List[List[Tuple[float, float]]], optional
        Prediction anchor points.
    clamp_max : float, optional
        Maximum clamp value for converting raw prediction to bounding box
        prediciton. This is useful for improving stability of the training.
        By default 5.0.
    """
    def __init__(self,
                 num_classes: int = 20,
                 anchors: List[List[Tuple]] = [
                     [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
                     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
                 ],
                 clamp_max: float = 5.0):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_scales, self.num_anchors, *_ = self.anchors.shape
        self.num_output = self.num_anchors * (5 + num_classes)
        self.num_classes = num_classes
        self.clamp_max = clamp_max
        self.scale = None

    def yolo(self, x: torch.tensor, anchors: torch.tensor) -> torch.tensor:
        """Evaluates YOLO bounding box prediction from raw network output.

        Parameters
        ----------
        x : torch.tensor
            Raw prediciton tensor.
        anchors : torch.tensor
            Anchors associated with the prediction head.

        Returns
        -------
        torch.tensor
            Output bounding boxes.
        """
        N, _, _, _, P, T = x.shape
        return _yolo(x, anchors, self.clamp_max).reshape([N, -1, P, T])

    def yolo_raw(self, x: torch.tensor) -> torch.tensor:
        """Transforms raw YOLO prediction to eventual output order i.e.
        NCHWT order to (batch, num_anchors, num_outputs, height, width, time).

        Parameters
        ----------
        x : torch.tensor
            Raw prediction output of the network.

        Returns
        -------
        torch.tensor
            Transformed raw prediction output for a head.
        """
        N, _, H, W, T = x.shape
        return x.reshape(N,
                         self.num_anchors,
                         -1, H, W, T).permute(0, 1, 3, 4, 2, 5)

    def init_model(self, input_dim: Tuple[int, int] = (448, 448)) -> None:
        """Initialize the network for a given input dimension.

        Parameters
        ----------
        input_dim : Tuple[int, int], optional
            Desired input dimension, by default (448, 448)
        """
        H, W = input_dim
        N, C, T = 1, 3, 1
        input = torch.rand(N, C, H, W, T).to(self.anchors.device)
        outputs = self.forward(input)[0]
        self.scale = [o.shape[2:4][::-1] for o in outputs]

    def validate_gradients(self) -> None:
        """Validate gradients of the network and circumvent if any.
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any()
                                       or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.zero_grad()
