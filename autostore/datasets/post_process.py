# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu


"""Post process model prediction results."""


import cv2
import numpy as np
from typing import Tuple, List


def magnify_single_bbox(
        org_bbox: np.ndarray,
        resize_scale: Tuple[float, float]
    ) -> None:
    h_ratio, w_ratio = resize_scale
    org_bbox[0] = int(np.round(org_bbox[0] / w_ratio))
    org_bbox[1] = int(np.round(org_bbox[1] / h_ratio))
    org_bbox[2] = int(np.round(org_bbox[2] / w_ratio))
    org_bbox[3] = int(np.round(org_bbox[3] / h_ratio))