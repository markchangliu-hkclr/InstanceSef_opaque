import os
import time
import cv2
import numpy as np
from pathlib import Path
from mmdet.apis import init_detector, inference_detector
from autostore.core.logging import get_logger
from autostore.core.io import load_img_paths


logger = get_logger(__name__)


def model_inference(
        img_dir: str,
        mmdet_cfg: str,
        weight_pth: str,
        show_score_thr: int,
        report_dir: str,
    ) -> None:
    model = init_detector(mmdet_cfg, weight_pth, device='cuda:0')
    time_deltas = []
    for img_path, img_name, img_ext in load_img_paths(img_dir):
        img = cv2.imread(img_path)
        t1 = time.time()
        result = inference_detector(model, img)
        t2 = time.time()
        time_deltas.append(t2 - t1)
        if report_dir:
            os.makedirs(Path(report_dir), exist_ok=True)
            visual_save_path = os.path.join(report_dir, f"{img_name}_inf.{img_ext}")
            model.show_result(
                img_path, 
                result, 
                score_thr=show_score_thr, 
                out_file=visual_save_path, 
                font_size=9
            )
    time_deltas = np.array(time_deltas)
    logger.info(f"avg time consumed for inferencing \
                {len(time_deltas)} images is {time_deltas.mean()}")
    

if __name__ == "__main__":
    img_dir = "/home/cliu/InstanceSef_opaque/data/commodity_group_rsz"
    mmdet_cfg = "/home/cliu/InstanceSef_opaque/configs/mmdet_commit/mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ_opaque.py"
    weight_pth = "/home/cliu/InstanceSef_opaque/exps/exps_opaque1/epoch_2_small.pth"
    show_score_thr = 0.8
    report_dir = "./exps/exps_opaque1_rsz"
    model_inference(img_dir, mmdet_cfg, weight_pth, show_score_thr, report_dir)