# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""MMDet library's model training, testing, etc. functions.
"""


import cv2
import numpy as np
import mmcv
import os
import torch
import time
from mmcv import Config
from mmdet.apis import set_random_seed, init_detector, inference_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from typing import Optional
from pathlib import Path
from autostore.core.logging import get_logger
from autostore.core.io import load_img_paths


_logger = get_logger(__name__)


def mmdet_model_train(mmdet_cfg: str, weight_path: str, output_dir: str) -> None:
    """Train a mmdet model with provided config file and pretrained weight,
    and output the trained weight to output_dir.

    Args:
        mmdet_cfg (str):
            MMDet config file path.
        weight_path (str):
            Pretrained weight path.
        output_dir (str):
            The path where trained weight will be save at.
    
    Returns:
        None
    """
    cfg = Config.fromfile(mmdet_cfg)
    cfg.load_from = weight_path
    _logger.info(f"Loaded config file from {mmdet_cfg}.")
    try:
        # This del operation is to avoid exception that exclusively happened 
        # on loading mask2former_r50 models
        del cfg.data.val.ins_ann_file
        del cfg.data.test.ins_ann_file
    except:
        pass
    cfg.device = "cuda"
    cfg.work_dir = output_dir
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = [0]
    
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=True, validate=True)
    
    try:
        checkpoint_name = '/epoch_'+str(cfg.runner.max_epochs)
    except:
        checkpoint_name = '/iter_'+str(cfg.runner.max_iters)
    checkpoint_path = os.path.join(cfg.work_dir, f"{checkpoint_name}.pth")
    checkpoint = torch.load(checkpoint_path)
    os.remove(checkpoint_path)
    meta = checkpoint['meta']
    checkpoint['optimizer'] = None
    weights = checkpoint['state_dict']
    state_dict = {"state_dict": weights, "meta": meta}
    new_checkpoint_path = os.path.join(cfg.work_dir, f"{checkpoint_name}_small.pth")
    torch.save(state_dict, new_checkpoint_path)


def mmdet_model_inference(
        img_dir: str,
        mmdet_cfg: str,
        weight_path: str,
        show_score_thr: float,
        report_dir: Optional[str] = None,
    ) -> None:
    """Perform inference on images with a mmdet model.
    
    Args:
        img_dir (str): 
            Image folder paths.
        mmdet_cfg (cfg):
            MMDet config file path.
        weight_path:
            Pretrained weight path.
        show_score_thr (float):
            BBox with score lower than this value will not show
            on inference result visualization.
        report_dir (Optional[str]):
            The path to save visualization results. if None, visualization
            results will not be saved.
    
    Returns:
        None
    """
    model = init_detector(mmdet_cfg, weight_path, device='cuda:0')
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
    _logger.info(f"avg time consumed for inferencing \
                {len(time_deltas)} images is {time_deltas.mean()}")
    return model, result