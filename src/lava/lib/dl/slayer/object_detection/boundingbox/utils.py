# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from PIL import ImageDraw
from PIL.Image import Image
from PIL import Image as Img

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, ops
from .metrics import bbox_iou

from typing import Tuple, List, Dict, Union, Iterable, Any, Optional

import cv2

RGB = int
Width = int
Height = int


def non_maximum_suppression(predictions: List[torch.tensor],
                            conf_threshold: float = 0.5,
                            nms_threshold: float = 0.4,
                            merge_conf: bool = True,
                            max_detections: int = 300,
                            max_iterations: int = 100) -> List[torch.tensor]:
    result = []
    for pred in predictions:
        filtered = pred[pred[:, 4] > conf_threshold]
        if not filtered.size(0):
            result.append(torch.zeros((0, 6), device=pred.device))
            continue

        boxes = filtered[:, :4]
        obj_conf, labels = torch.max(filtered[:, 5:], dim=1, keepdim=True)
        if merge_conf:
            scores = filtered[:, 4:5] * obj_conf
        else:
            scores = filtered[:, 4:5]

        order = torch.argsort(scores.squeeze(), descending=True)
        # Custon NMS loop
        detections = torch.cat([boxes, scores, labels], dim=-1)
        prev_objects = detections.shape[0]
        if order.shape:
            detections = detections[order]
            for i in range(max_iterations):
                ious = bbox_iou(detections, detections)
                label_match = (
                    detections[:, 5].reshape(-1, 1)
                    == detections[:, 5].reshape(1, -1)
                ).long().view(ious.shape)

                keep = (
                    ious * label_match > nms_threshold
                ).long().triu(1).sum(dim=0,
                                     keepdim=True).T.expand_as(detections) == 0

                detections = detections[keep].reshape(-1, 6).contiguous()
                if detections.shape[0] == prev_objects:
                    break
                else:
                    prev_objects = detections.shape[0]
        # #
        # # above gives slightly better scores
        # idx = ops.nms(ops._box_convert._box_xywh_to_xyxy(boxes.clone()),
        #               scores.flatten(), nms_threshold)
        # detections = torch.cat([boxes[idx], scores[idx], labels[idx]], dim=-1)

        if detections.shape[0] > max_detections:
            detections = detections[:max_detections]
        result.append(detections)
    return result


def yolo_loss():
    # move to yolo_base.py
    pass


def annotation_from_tensor(tensor: torch.tensor,
                           frame_size: Dict[str, int],
                           object_names: Iterable[str],
                           confidence_th: float = 0.01,
                           normalized: bool = True) -> Dict[str, Any]:
    annotation = {'annotation': {'size': frame_size}}
    if normalized:
        height = frame_size['height']
        width = frame_size['width']
    else:
        height = 1
        width = 1

    objects = []

    boxes = tensor[..., :4].reshape((-1, 4)).cpu().data.numpy()
    xmin = (boxes[:, 0] - boxes[:, 2] / 2) * width
    ymin = (boxes[:, 1] - boxes[:, 3] / 2) * height
    xmax = (boxes[:, 0] + boxes[:, 2] / 2) * width
    ymax = (boxes[:, 1] + boxes[:, 3] / 2) * height
    if boxes.shape[0] > 0:
        confidences = tensor[..., 4].flatten().cpu().data.numpy()
        class_labels = tensor[..., 5].flatten().cpu().data.numpy()
        for i in range(boxes.shape[0]):
            confidence = confidences[i].item()
            if confidence < confidence_th:
                continue
            class_idx = int(class_labels[i].item())

            objects += [{
                'id': class_idx,
                'name': object_names[class_idx],
                'confidence': confidence,
                'bndbox': {'xmin': xmin[i].item(), 'ymin': ymin[i].item(),
                        'xmax': xmax[i].item(), 'ymax': ymax[i].item()}
            }]

    annotation['annotation']['object'] = objects

    return annotation


def tensor_from_annotation(ann: Dict[str, Any],
                           device: torch.device = torch.device('cpu'),
                           num_objects: Optional[int] = None,
                           normalized: bool = True) -> torch.tensor:
    if normalized:
        height = int(ann['annotation']['size']['height'])
        width = int(ann['annotation']['size']['width'])
    else:
        height = 1
        width = 1
    boxes = []
    area = []
    for object in ann['annotation']['object']:
        if 'confidence' in object.keys():
            confidence = object['confidence']
        else:
            confidence = 1.0
        xmin = object['bndbox']['xmin']
        xmax = object['bndbox']['xmax']
        ymin = object['bndbox']['ymin']
        ymax = object['bndbox']['ymax']
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([(xmin + xmax) / width / 2,
                      (ymin + ymax) / height / 2,
                      (xmax - xmin) / width,
                      (ymax - ymin) / height,
                      confidence,  # confidence
                      object['id']])  # label
        area.append((xmax - xmin) * (ymax - ymin))

    idx = np.argsort(area)[::-1]
    boxes = torch.FloatTensor([boxes[i] for i in idx], device=device)
    if num_objects:
        if num_objects < len(boxes):
            boxes = boxes[:num_objects]
        # elif num_objects > len(boxes):
        #     boxes = torch.cat([boxes,
        #                        torch.zeros((num_objects - len(boxes,
        #                                     6))).to(device)], dim=0)

    return boxes


def onehot_to_labels(predictions: List[torch.tensor],
                     conf_threshold: float = 0.0,
                     merge_conf: bool = True) -> List[torch.tensor]:
    new_predictions = []
    for pred in predictions:
        conf, label = torch.max(pred[:, 5:], axis=1)
        if not merge_conf:
            conf = torch.ones_like(conf)
        pred = torch.cat([pred[:, :4],
                          pred[:, 4:5] * conf.view(-1, 1),
                          label.view(-1, 1)], axis=1)
        new_predictions.append(pred[pred[:, 4] > conf_threshold])
    return new_predictions


def xxyy_to_xywh(predictions: List[torch.tensor]) -> List[torch.tensor]:
    translated = []
    for p in predictions:
        pred = p.clone()
        xmin = pred[:, 0].clone()
        ymin = pred[:, 1].clone()
        xmax = pred[:, 2].clone()
        ymax = pred[:, 3].clone()
        pred[:, 0] = (xmin + xmax) / 2
        pred[:, 1] = (ymin + ymax) / 2
        pred[:, 2] = (xmax - xmin)
        pred[:, 3] = (ymax - ymin)
        translated.append(pred)
    return translated


def xywh_to_xxyy(predictions: List[torch.tensor]) -> List[torch.tensor]:
    translated = []
    for p in predictions:
        pred = p.clone()
        xmin = pred[:, 0].clone()
        ymin = pred[:, 1].clone()
        width = pred[:, 2].clone()
        height = pred[:, 3].clone()
        pred[:, 2] = (xmin + width)
        pred[:, 3] = (ymin + height)
        translated.append(pred)
    return translated


def normalize_bboxes(predictions: List[torch.tensor],
                     height: Height, width: Width) -> List[torch.tensor]:
    for idx in range(len(predictions)):
        predictions[idx][:, [0, 2]] /= width
        predictions[idx][:, [1, 3]] /= height
    return predictions


def denormalize_bboxes(predictions: List[torch.tensor],
                       height: Height, width: Width) -> List[torch.tensor]:
    for idx in range(len(predictions)):
        predictions[idx][:, [0, 2]] *= width
        predictions[idx][:, [1, 3]] *= height
    return predictions


def merge_annotations(ann0: Dict[str, Any],
                      ann1: Dict[str, Any]) -> Dict[str, Any]:
    if (
        ann0['annotation']['size']['height']
        != ann1['annotation']['size']['height']
    ):
        raise ValueError(f'Annotation dimension do not match! '
                         f"found {ann0['annotation']['size']=} and "
                         f"found {ann1['annotation']['size']=} and ")

    if (
        ann0['annotation']['size']['width']
        != ann1['annotation']['size']['width']
    ):
        raise ValueError(f'Annotation dimension do not match! '
                         f"found {ann0['annotation']['size']=} and "
                         f"found {ann1['annotation']['size']=} and ")

    return {'annotation': {'size': ann0['annotation']['size']},
            'object': ann0['annotation']['object']
            + ann1['annotation']['object']}


def mark_bounding_boxes(
    image: Union[Image, torch.tensor],
    objects: Dict[str, Any],
    box_color_map: List[Tuple[RGB, RGB, RGB]] = [],  # this takes precedence
    box_color: Tuple[RGB, RGB, RGB] = (0, 255, 0),   # over box_color
    text_color: Tuple[RGB, RGB, RGB] = (0, 0, 255),  # and text_color
    thickness: int = 5
) -> Image:
    if torch.is_tensor(image):
        image = transforms.ToPILImage()(image.squeeze())
    draw = ImageDraw.Draw(image)
    for object in objects:
        name = object['name']
        bndbox = object['bndbox']
        if 'confidence' in object.keys():
            confidence = object['confidence']
        else:
            confidence = 1
        box = [max(bndbox['xmin'], 0),
               max(bndbox['ymin'], 0),
               min(bndbox['xmax'], image.width - 1),
               min(bndbox['ymax'], image.height - 1)]
        box = [int(x) for x in box]
        if box_color_map:
            box_color = box_color_map[int(object['id']) % len(box_color_map)]
            text_color = tuple(255 - c for c in box_color)
        draw.rectangle(box, outline=box_color,
                       width=max(int(thickness * confidence), 1))
        draw.rectangle([box[0], box[1], box[0] + 40, box[1] + 10],
                       outline=box_color, fill=box_color)
        draw.text(box[:2], name, fill=text_color)
    return image


def resize_bounding_boxes(annotation: Dict[str, Any],
                          size: Tuple[Height, Width]) -> Dict[str, Any]:
    ann = annotation
    height = int(ann['annotation']['size']['height'])
    width = int(ann['annotation']['size']['width'])

    def height_fx(x):
        return x / height * size[0]

    def width_fx(x):
        return x / width * size[1]

    ann['annotation']['size']['width'] = width_fx(width)
    ann['annotation']['size']['height'] = height_fx(height)
    for i in range(len(ann['annotation']['object'])):
        xmin = int(ann['annotation']['object'][i]['bndbox']['xmin'])
        ymin = int(ann['annotation']['object'][i]['bndbox']['ymin'])
        xmax = int(ann['annotation']['object'][i]['bndbox']['xmax'])
        ymax = int(ann['annotation']['object'][i]['bndbox']['ymax'])
        ann['annotation']['object'][i]['bndbox']['xmin'] = width_fx(xmin)
        ann['annotation']['object'][i]['bndbox']['ymin'] = height_fx(ymin)
        ann['annotation']['object'][i]['bndbox']['xmax'] = width_fx(xmax)
        ann['annotation']['object'][i]['bndbox']['ymax'] = height_fx(ymax)
    return annotation


def flipud_bounding_boxes(annotation: Dict[str, Any]) -> Dict[str, Any]:
    ann = annotation
    height = int(ann['annotation']['size']['height'])

    def height_fx(x):
        return height - x

    ann['annotation']['size']['height'] = height
    for i in range(len(ann['annotation']['object'])):
        ymin = int(ann['annotation']['object'][i]['bndbox']['ymin'])
        ymax = int(ann['annotation']['object'][i]['bndbox']['ymax'])
        ann['annotation']['object'][i]['bndbox']['ymin'] = height_fx(ymax)
        ann['annotation']['object'][i]['bndbox']['ymax'] = height_fx(ymin)
    return annotation


def fliplr_bounding_boxes(annotation: Dict[str, Any]) -> Dict[str, Any]:
    ann = annotation
    width = int(ann['annotation']['size']['width'])

    def width_fx(x):
        return width - x

    ann['annotation']['size']['width'] = width
    for i in range(len(ann['annotation']['object'])):
        xmin = int(ann['annotation']['object'][i]['bndbox']['xmin'])
        xmax = int(ann['annotation']['object'][i]['bndbox']['xmax'])
        ann['annotation']['object'][i]['bndbox']['xmin'] = width_fx(xmax)
        ann['annotation']['object'][i]['bndbox']['xmax'] = width_fx(xmin)
    return annotation

def create_video(inputs, targets, predictions, video_output_path, BOX_COLOR_MAP, classes ):
    b = 0
    video_dims = (2 * 448, 448)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_output_path + '.mp4', fourcc, 10, video_dims)
    for t in range(inputs.shape[-1]):
        image = Img.fromarray(np.uint8(inputs[b, :, :, :, t].cpu().data.numpy().transpose([1, 2, 0]) * 255))
        annotation = annotation_from_tensor(predictions[t][b],
                                            {'height': image.height,
                                                'width': image.width},
                                            classes,
                                            confidence_th=0)
        marked_img = mark_bounding_boxes(image, annotation['annotation']['object'],
                                            box_color_map=BOX_COLOR_MAP, thickness=5)
        image = Img.fromarray(np.uint8(inputs[b, :, :, :, t].cpu().data.numpy().transpose([1, 2, 0]) * 255))
        annotation = annotation_from_tensor(targets[t][b],
                                            {'height': image.height,
                                                'width': image.width},
                                            classes,
                                            confidence_th=0)
        marked_gt = mark_bounding_boxes(image, annotation['annotation']['object'],
                                        box_color_map=BOX_COLOR_MAP, thickness=5)
        marked_images = Img.new('RGB', (marked_img.width + marked_gt.width, marked_img.height))
        marked_images.paste(marked_img, (0, 0))
        marked_images.paste(marked_gt, (marked_img.width, 0))
        video.write(cv2.cvtColor(np.array(marked_images), cv2.COLOR_RGB2BGR))
    video.release()

