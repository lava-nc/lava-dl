# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import Iterable, Optional, List, Callable
from queue import Queue
import numpy as np
import torch

from lava.lib.dl.netx.sequential_modules import AbstractSeqModule
from lava.lib.dl.slayer import obd


class DataGenerator(AbstractSeqModule):
    """Datagenerator module for Object detection. On every call, it
    returns an RGB frame (with normalization applied), ground truth
    annotation, and the raw frame.

    Next sample is automatically loaded when the buffer is empty.

    Parameters
    ----------
    dataset : Iterable
        Object detection dataset
    start_idx : int, optional
        Desired starting index, by default 0.
    mean : Optional[np.ndarray], optional
        Normalization mean, by default None.
    std : Optional[np.ndarray], optional
        Normalization standard deviation, by default None.
    """
    def __init__(self,
                 dataset: Iterable,
                 start_idx: int = 0,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.sample_idx = start_idx
        self.frame_idx = 0
        frames, annotations = dataset[self.sample_idx]
        self.frames = frames
        self.annotations = annotations
        self.sample_idx += 1
        self.mean = mean
        self.std = std
        self.normalize = mean is not None

    def __call__(self) -> None:
        return super().__call__()

    def forward(self) -> None:
        raw_frame = self.frames[..., self.frame_idx]
        frame = raw_frame.clone().permute([2, 1, 0])  # CHW to XYZ
        annotation = self.annotations[self.frame_idx]
        self.frame_idx += 1
        if self.normalize:
            frame = (frame - self.mean) / self.std
        return frame, annotation, raw_frame

    def post_forward(self) -> None:
        if self.frame_idx >= len(self.annotations):
            self.frame_idx = 0
            frames, annotations = self.dataset[self.sample_idx]
            self.frames = frames
            self.annotations = annotations
            self.sample_idx += 1


class YOLOPredictor(AbstractSeqModule):
    """YOLO prediction process. This expects numpy data in XYZ format unlike
    the similar module used in `slayer.obd`.

    On every call, it takes a raw 3D tensor in XYF format and returns
    the YOLO num_anchors * X * Y bounding box predictions in the order of
    x_center, y_center, width, height, confidence and one-hot classes.

    Parameters
    ----------
    anchors : np.ndarray
        Anchor prediction references.
    clamp_max : float, optional
        Maximum clamp values during exponentiation, by default 5.0.
    """
    def __init__(self,
                 anchors: np.ndarray,
                 clamp_max: float = 5.0) -> None:
        super().__init__()
        self.anchors = anchors
        self.clamp_max = clamp_max
        self.num_anchors = anchors.shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return super().__call__(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        W, H, C = x.shape
        P = C // self.num_anchors
        x = x.reshape([W, H, self.num_anchors, P]).transpose([2, 1, 0, 3])

        range_y, range_x = np.meshgrid(np.arange(H), np.arange(W),
                                       indexing='ij')

        anchor_x, anchor_y = self.anchors[:, 0], self.anchors[:, 1]
        x_center = (sigmoid(x[:, :, :, 0:1]) + range_x[None, :, :, None]) / W
        y_center = (sigmoid(x[:, :, :, 1:2]) + range_y[None, :, :, None]) / H
        width = (np.exp(x[:, :, :, 2:3].clip(max=self.clamp_max))
                 * anchor_x[:, None, None, None]) / W
        height = (np.exp(x[:, :, :, 3:4].clip(max=self.clamp_max))
                  * anchor_y[:, None, None, None]) / H
        confidence = sigmoid(x[:, :, :, 4:5])
        classes = softmax(x[:, :, :, 5:], axis=-1)

        x = np.concatenate([x_center, y_center,
                            width, height,
                            confidence, classes], axis=-1)

        x = x.reshape([-1, P])

        return x


class YOLOMonitor(AbstractSeqModule):
    def __init__(self, class_list: List[str],
                 viz_fx: Optional[Callable] = None) -> None:
        """YOLO output monitor. This module is responsible for displaying
        the output predictions and evaluating the mAP performance of the
        network.

        On every call, it evaluates the network output (mAP) and creates
        prediction frames.

        Parameters
        ----------
        class_list : List[str]
            List of calss id in the dataset.
        viz_fx : Optional[Callable]
            Output visualization function. It is expected to take three args:
            annotated_frame (PIL Image), map_score (float), frame_idx (int).
            If it is None, no visualization is done.
        """
        super().__init__()
        self.ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)
        self.input_frame_buffer = Queue()
        self.gt_bbox_buffer = Queue()
        self.pred_bbox_buffer = Queue()
        self.class_list = class_list
        self.viz_fx = viz_fx
        self.box_color_map = [(np.random.randint(256),
                               np.random.randint(256),
                               np.random.randint(256))
                              for _ in range(len(class_list))]

    def __call__(self,
                 input_frame: torch.tensor,
                 gt_bbox: torch.tensor,
                 pred_bbox: torch.tensor) -> None:
        return super().__call__(input_frame, gt_bbox, pred_bbox)

    def forward(self,
                input_frame: torch.tensor,
                gt_bbox: torch.tensor,
                pred_bbox: torch.tensor) -> None:
        self.input_frame_buffer.put(input_frame)
        self.gt_bbox_buffer.put(gt_bbox)
        self.pred_bbox_buffer.put(pred_bbox)

    def post_forward(self) -> None:
        input_frame = self.input_frame_buffer.get()
        gt_bbox = self.gt_bbox_buffer.get()
        pred_bbox = self.pred_bbox_buffer.get()
        self.ap_stats.update([torch.tensor(pred_bbox)],
                             [torch.tensor(gt_bbox)])
        annotated_frame = obd.bbox.utils.create_frames(
            inputs=input_frame[None, ..., None],
            targets=[[torch.tensor(gt_bbox)]],
            predictions=[[torch.tensor(pred_bbox)]],
            classes=self.class_list,
            box_color_map=self.box_color_map)[0]
        if self.viz_fx is not None:
            self.viz_fx(annotated_frame, self.ap_stats[:], self.time_step)


# Wrapped function calls
def nms(predictions: np.ndarray,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        merge_conf: bool = True,
        max_detections: int = 300,
        max_iterations: int = 100) -> np.ndarray:
    return obd.boundingbox.utils.nms([torch.tensor(predictions)],
                                     conf_threshold,
                                     nms_threshold,
                                     merge_conf,
                                     max_detections,
                                     max_iterations)[0].cpu().data.numpy()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis=None) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
