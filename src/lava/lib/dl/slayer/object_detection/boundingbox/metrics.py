# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple, Union

import numpy as np
import torch

"""Object detection metrics."""

# Infinitesimal for floating point relaxation
EPS = 1e-10


def bbox_iou(bbox1: torch.tensor, bbox2: torch.tensor) -> torch.tensor:
    """Evaluates the intersection over union (IOU) over two sets of bounding
    boxes tensors.

    Parameters
    ----------
    bbox1 : torch.tensor
        Bounding box tensor in format (x_center, y_center, width, height).
    bbox2 : torch.tensor
        Bounding box tensor in format (x_center, y_center, width, height).

    Returns
    -------
    torch.tensor
        IOU tensor.
    """
    xmin1 = (bbox1[..., 0] - 0.5 * bbox1[..., 2]).reshape((1, -1))
    xmax1 = (bbox1[..., 0] + 0.5 * bbox1[..., 2]).reshape((1, -1))
    ymin1 = (bbox1[..., 1] - 0.5 * bbox1[..., 3]).reshape((1, -1))
    ymax1 = (bbox1[..., 1] + 0.5 * bbox1[..., 3]).reshape((1, -1))

    xmin2 = (bbox2[..., 0] - 0.5 * bbox2[..., 2]).reshape((1, -1))
    xmax2 = (bbox2[..., 0] + 0.5 * bbox2[..., 2]).reshape((1, -1))
    ymin2 = (bbox2[..., 1] - 0.5 * bbox2[..., 3]).reshape((1, -1))
    ymax2 = (bbox2[..., 1] + 0.5 * bbox2[..., 3]).reshape((1, -1))

    zero = torch.FloatTensor([0]).to(bbox1.dtype).to(bbox1.device)
    dx = torch.max(torch.min(xmax1, xmax2.T) - torch.max(xmin1, xmin2.T), zero)
    dy = torch.max(torch.min(ymax1, ymax2.T) - torch.max(ymin1, ymin2.T), zero)

    intersections = dx * dy
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    unions = (area1 + area2.T) - intersections + EPS
    ious = (intersections / unions).reshape(*bbox1.shape[:-1],
                                            *bbox2.shape[:-1])

    return ious


def wh_iou(bbox1: torch.tensor, bbox2: torch.tensor) -> torch.tensor:
    """Evaluates the intersection over union (IOU) only based on widht and
    height information assuming maximum overlap.

    Parameters
    ----------
    bbox1 : torch.tensor
        Bounding box tensor in format (width, height).
    bbox2 : torch.tensor
        Bounding box tensor in format (width, height).

    Returns
    -------
    torch.tensor
        Width-Height IOU tensor.
    """
    w1 = bbox1[..., 0]
    h1 = bbox1[..., 1]
    w2 = bbox2[..., 0]
    h2 = bbox2[..., 1]
    intersections = (torch.min(w1[:, None], w2[None, :])
                     * torch.min(h1[:, None], h2[None, :]))
    unions = (w1 * h1)[:, None] + (w2 * h2)[None, :] - intersections + EPS
    ious = (intersections / unions).reshape(*bbox1.shape[:-1],
                                            *bbox2.shape[:-1])
    return ious


def bbox_ciou(bbox1: torch.tensor, bbox2: torch.tensor) -> torch.tensor:
    """Evaluates differentiable form of intersection over union
    (Complete IOU loss) based on distance between centers as described in
    https://arxiv.org/abs/1911.08287v1

    Parameters
    ----------
    bbox1 : torch.tensor
        Bounding box tensor in format (x_center, y_center, width, height).
    bbox2 : torch.tensor
        Bounding box tensor in format (x_center, y_center, width, height).

    Returns
    -------
    torch.tensor
        C-IOU tensor.
    """
    # https://arxiv.org/abs/1911.08287v1
    xmin1 = (bbox1[..., 0] - 0.5 * bbox1[..., 2]).flatten()
    xmax1 = (bbox1[..., 0] + 0.5 * bbox1[..., 2]).flatten()
    ymin1 = (bbox1[..., 1] - 0.5 * bbox1[..., 3]).flatten()
    ymax1 = (bbox1[..., 1] + 0.5 * bbox1[..., 3]).flatten()

    xmin2 = (bbox2[..., 0] - 0.5 * bbox2[..., 2]).flatten()
    xmax2 = (bbox2[..., 0] + 0.5 * bbox2[..., 2]).flatten()
    ymin2 = (bbox2[..., 1] - 0.5 * bbox2[..., 3]).flatten()
    ymax2 = (bbox2[..., 1] + 0.5 * bbox2[..., 3]).flatten()

    zero = torch.FloatTensor([0]).to(bbox1.dtype).to(bbox1.device)
    dx = torch.max(torch.min(xmax1, xmax2) - torch.max(xmin1, xmin2), zero)
    dy = torch.max(torch.min(ymax1, ymax2) - torch.max(ymin1, ymin2), zero)

    intersections = dx * dy
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    unions = (area1 + area2) - intersections + EPS
    ious = intersections / unions

    # smallest enclosing box between two regions
    cx = torch.max(xmax1, xmax2) - torch.min(xmin1, xmin2)
    cy = torch.max(ymax1, ymax2) - torch.min(ymin1, ymin2)

    # diagonal length of smallest enclosing box
    c2 = cx**2 + cy**2 + EPS

    # distance between center points of two regions
    rho2 = ((xmax2 + xmin2 - xmax1 - xmin2)**2
            + (ymax2 + ymin2 - ymax1 - ymin2)**2) / 4

    # consistency of aspect ratio term
    v = (4 / np.pi**2) * torch.pow(torch.atan((xmax1 - xmin1)
                                              / (ymax1 - ymin1 + EPS))
                                   - torch.atan((xmax2 - xmin2)
                                                / (ymax2 - ymin2 + EPS)), 2)
    # trade off parameter
    with torch.no_grad():
        alpha = v / (1 - ious + v + EPS)

    return ious - (rho2 / c2 + alpha * v)


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """Evaluates Average Precision metric (area under the precision recall
    curve) based on precision and recall data points measured. This method
    uses trapezoid method to integrate the area.

    Parameters
    ----------
    precision : np.ndarray
        Precision measurement points.
    recall : np.ndarray
        Recall measurement points.

    Returns
    -------
    float
        Average Precision value.
    """
    recall = np.concatenate(([1.], recall, [recall[-1] + 0.01]))
    precision = np.concatenate(([1.], precision, [0.]))
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))
    x = np.linspace(0, 1, 101)  # 101 point interpolation (COCO)
    ap = np.trapz(np.interp(x, recall, precision), x)
    return ap


def average_precision_metrics(
    outputs: List[torch.tensor],
    targets: List[torch.tensor],
    iou_threshold: Union[float,
                         np.ndarray,
                         List[float]]
) -> Tuple[List[np.ndarray],  # precision
           List[np.ndarray],  # recall
           List[np.ndarray],  # AP
           List[np.ndarray],  # f1
           List[np.ndarray],  # unique classes
           ]:
    """Evaluates average precision metrics from the output and target
    bounding boxes for each of the IOU threshold points. In addition,
    it also returns the precision, recall, F1 scores and the unique
    classes in target labels. It expectes list of bonunding boxes in
    a batch.

    Parameters
    ----------
    outputs : list of torch.tensor
        List of output bounding box prediction tensor for every batch.
    targets : list of torch.tensor
        List of output bounding box prediction tensor for every batch.
    iou_threshold : float or np.ndarray or list of floats
        IOU threshold(s) for a prediction to be considered true positive.

    Returns
    -------
    list of np.ndarray
        Precision score for each batch.
    list of np.ndarray
        Recall score for each batch.
    list of np.ndarray
        Average precision score for each batch.
    list of np.ndarray
        F1 score for each batch.
    list of np.ndarray
        Unique classes for each batch.
    """
    num_iou = 1 if np.isscalar(iou_threshold) else len(iou_threshold)
    precision_list = []
    recall_list = []
    average_precision_list = []
    f1_score_list = []
    unique_classes_list = []
    for batch in range(len(outputs)):
        if outputs[batch] is None:
            continue
        output = outputs[batch]
        pred_boxes = output[:, :4].cpu()
        pred_conf = output[:, 4].cpu().data.numpy()
        pred_labels = output[:, -1].cpu().data.numpy()

        true_positives = np.zeros((pred_boxes.shape[0], num_iou))

        target = targets[batch]
        if len(target) == 0:
            continue
        target_boxes = target[:, :4].cpu()
        target_labels = target[:, -1].cpu().data.numpy()

        ious = bbox_iou(pred_boxes, target_boxes)
        detected_boxes = []
        for pred_idx, (iou, pred_label) in enumerate(zip(ious, pred_labels)):
            if len(detected_boxes) == len(target):
                break

            if pred_label not in target_labels:
                continue

            iou, target_idx = iou.unsqueeze(0).max(axis=1)

            if target_idx not in detected_boxes:
                true_positives[pred_idx] = iou.item() > iou_threshold
                detected_boxes += [target_idx]

        i = np.argsort(-pred_conf)
        true_positives = true_positives[i]
        pred_conf = pred_conf[i]
        pred_labels = pred_labels[i]

        unique_classes = np.unique(target_labels)
        average_precision, precision, recall = [], [], []
        px, py = np.linspace(0, 1, 1000), []
        for c in unique_classes:
            i = pred_labels == c
            num_ground_truth = (target_labels == c).sum()
            num_predicted = i.sum()

            if num_predicted == 0 and num_ground_truth == 0:
                continue
            elif num_predicted == 0 or num_ground_truth == 0:
                precision.append(np.zeros((num_iou,)))
                recall.append(np.zeros((num_iou,)))
                average_precision.append(np.zeros((num_iou,)))
            else:
                false_positives_cum = (1 - true_positives[i]).cumsum(axis=0)
                true_positives_cum = (true_positives[i]).cumsum(axis=0)

                recall_curve = true_positives_cum / (num_ground_truth + EPS)
                recall.append(recall_curve[-1])

                precision_curve = true_positives_cum / \
                    (true_positives_cum + false_positives_cum)
                precision.append(precision_curve[-1])

                ap = np.zeros((num_iou,))
                for ii in range(num_iou):
                    ap[ii] = compute_ap(precision_curve[:, ii],
                                        recall_curve[:, ii])
                average_precision.append(ap)

        precision = np.array(precision)
        recall = np.array(recall)
        average_precision = np.array(average_precision)
        f1_score = precision * recall / (precision + recall + EPS)

        precision_list.append(precision)
        recall_list.append(recall)
        average_precision_list.append(average_precision)
        f1_score_list.append(f1_score)
        unique_classes_list.append(unique_classes)

    return (
        precision_list,
        recall_list,
        average_precision_list,
        f1_score_list,
        unique_classes_list,
    )


class APstats:
    """Average Prcision stats manager. It helps collecting mean average
    precision for each batch predictions and targets and summarize
    the result.

    Parameters
    ----------
    iou_threshold : Union[float, np.ndarray, List[float]]
        IOU threshold(s) for a prediction to be considered true positive.
    """
    def __init__(self,
                 iou_threshold: Union[float,
                                      np.ndarray,
                                      List[float]]) -> None:
        self.num_iou = 1 if np.isscalar(iou_threshold) else len(iou_threshold)
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        """Reset mAP statistics.
        """
        self.ap_sum = np.zeros(self.num_iou)
        self.count = 0

    def update(self,
               predictions: List[torch.tensor],
               targets: List[torch.tensor]) -> None:
        """Update the mAP statistics.

        Parameters
        ----------
        predictions : List[torch.tensor]
            List of prediction tensors for every batch.
        targets : List[torch.tensor]
            List of target tensors for every batch.
        """
        ap_metrics = average_precision_metrics(predictions,
                                               targets,
                                               self.iou_threshold)[2]
        if len(ap_metrics) > 0:
            ap = np.concatenate(ap_metrics)
            self.ap_sum += np.sum(ap, axis=0)
            self.count += ap.shape[0]

    def ap_scores(self) -> np.ndarray:
        """Evaluate mAP scores for all of the IOU thresholds.

        Returns
        -------
        np.ndarray
            mAP score(s)
        """
        if self.count == 0:
            return self.ap_sum
        return self.ap_sum / self.count

    def __getitem__(self, iou: Union[float, int, slice]) -> float:
        """Returns selected mAP score. The mAP scores can be addressed baed on
        IOU threshold indices, IOU threshold values or a slice. Slice will
        evaluate the aggregate IOU scores suitable for scores like
        ```AP[:] = mAP@{all_iou_thresholds}```.

        Parameters
        ----------
        iou : Union[float, int, slice]
            If float, it is the IOU value to index; if int, it is the AP
            corresponding to the IOU index; if slice, it is the aggregrate sum
            of all IOU threshold values.

        Returns
        -------
        float
            mAP values.
        """
        if iou == slice(None, None, None):
            return np.mean(self.ap_scores()).item()
        elif type(iou) == float:
            if iou in list(self.iou_threshold):
                idx = np.argwhere(self.iou_threshold == iou)[0]
                return self.ap_scores()[idx].item()
            else:
                RuntimeError(f'Query IOU threshold is not recoreded. '
                             f'Try one of {self.iou_threshold},')
        return self.ap_scores()[iou]
