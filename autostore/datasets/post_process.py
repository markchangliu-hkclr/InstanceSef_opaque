# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu


"""Post process model prediction results."""


import cv2
import numpy as np
import PIL.Image as pil_image
import torchvision.transforms.functional as F
from autostore.datasets.transforms import crop_single_img
from sklearn.metrics import jaccard_score
from typing import Tuple, List, Any


def compute_iou(
        bbox_a: np.ndarray,
        bbox_b: np.ndarray
    ) -> float:
    x1_a, y1_a, x2_a, y2_a = bbox_a[0, :4]
    x1_b, y1_b, x2_b, y2_b = bbox_b[0, :4]
    x1_int = max(x1_a, x1_b)
    y1_int = max(y1_a, y1_b)
    x2_int = min(x2_a, x2_b)
    y2_int = min(y2_a, y2_b)
    int_area = (x2_int - x1_int) * (y2_int - y1_int)
    bbox_a_area = (x2_a - x1_a) * (y2_a - y1_a)
    bbox_b_area = (x2_b - x1_b) * (y2_b - y2_a)
    return int_area / (bbox_a_area + bbox_b_area - int_area)


def restore_single_bbox(
        new_bbox_: np.ndarray,
        rsz_size: Tuple[int, int],
        img_size: Tuple[int, int]
    ) -> np.ndarray:
    new_bbox = np.copy(new_bbox_)
    img_h, img_w = img_size
    rsz_h, rsz_w = rsz_size
    pad_top = (img_h - rsz_h) // 2
    pad_bottom = img_h - rsz_h - pad_top
    pad_left = (img_w - rsz_w) // 2
    pad_right = img_w - rsz_w - pad_left
    new_x1, new_y1, new_x2, new_y2, score = new_bbox
    new_x1_unpad = new_x1 - pad_left
    new_y1_unpad = new_y1 - pad_top
    new_x2_unpad = new_x2 - pad_left
    new_y2_unpad = new_y2 - pad_top
    org_x1 = new_x1_unpad * (img_w / rsz_w)
    org_y1 = new_y1_unpad * (img_h / rsz_h)
    org_x2 = new_x2_unpad * (img_w / rsz_w)
    org_y2 = new_y2_unpad * (img_h / rsz_h)
    org_bbox = np.array([org_x1, org_y1, org_x2, org_y2, score])
    return org_bbox[np.newaxis, :]


def restore_bboxes(
        new_bboxes: List[np.ndarray],
        rsz_size: Tuple[int, int],
        img_size: Tuple[int, int],
    ) -> List[np.ndarray]:
    org_bboxes = []
    for new_bbox in new_bboxes:
        org_bbox = restore_single_bbox(
            new_bbox, 
            rsz_size, 
            img_size
        )
        org_bboxes.append(org_bbox)
    return org_bboxes


def remove_irregular_bboxes(
        bboxes: List[np.ndarray],
        img_size: Tuple[int, int],
        iou_thr: float
    ) -> List[int]:
    img_h, img_w = img_size
    bbox_img = np.array([[0, 0, img_w, img_h, 1]])
    valid_box_idx = []
    for i, bbox in enumerate(bboxes):
        iou = compute_iou(bbox, bbox_img)
        if iou < iou_thr:
            valid_box_idx.append(i)
    return valid_box_idx


def restore_seg_mask(
        seg_masks: List[np.ndarray],
        rsz_size: Tuple[int, int],
        img_size: Tuple[int, int]
    ) -> List[np.ndarray]:
    org_seg_masks = []
    for seg_mask in seg_masks:
        seg_mask_img = pil_image.fromarray(seg_mask)
        org_seg_mask = F.resize(seg_mask_img, img_size)
        org_seg_mask = np.asarray(org_seg_mask)
        org_seg_masks.append(org_seg_mask)
    return org_seg_masks


def remove_irregular_seg_masks(
        seg_masks: List[np.ndarray],
        img_size: Tuple[int, int],
        iou_thr: float
    ) -> List[int]:
    valid_seg_mask_idx = []
    seg_mask_img = np.ones(img_size).astype(np.bool8)
    for i, seg_mask in enumerate(seg_masks):
        iou = jaccard_score(seg_mask_img, seg_mask, average="samples")
        if iou < iou_thr:
            valid_seg_mask_idx.append(i)
    return valid_seg_mask_idx