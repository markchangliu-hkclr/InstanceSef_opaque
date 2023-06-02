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
import PIL.Image as pil_image
import torch
import time
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import set_random_seed, init_detector, inference_detector
from mmdet.core import encode_mask_results
from mmdet.datasets import (
    build_dataset,
    build_dataloader,
    replace_ImageToTensor
)
from mmdet.models import build_detector
from mmdet.apis import train_detector, single_gpu_test
from typing import Optional, Tuple, List, NewType, Dict, Any, Callable
from pathlib import Path
from PIL.Image import Image
from autostore.core.logging import get_logger
from autostore.core.io import load_img_paths
from autostore.datasets.pre_process import (
    rsz_imgs_from_csv_info,
    rsz_paste_imgs_from_csv_info
)
from autostore.datasets.post_process import (
    restore_bboxes,
    restore_seg_mask,
    remove_irregular_bboxes,
    remove_irregular_seg_masks
)


_logger = get_logger(__name__)


EvaluateResultType = NewType(
    "EvaluateResultType",
    List[Tuple[List[np.ndarray], List[List[Dict[str, Any]]]]]
)


def train_model(
        mmdet_cfg: str, 
        weight_path: str, 
        output_dir: str
    ) -> None:
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


def test_imgs(
        mmdet_cfg: str,
        weight_path: str, 
        show_score_thr: float,
        report_dir: str,
    ) -> None:
    cfg = Config.fromfile(mmdet_cfg)
    try:
        # This del operation is to avoid exception that exclusively happened 
        # on loading mask2former_r50 models
        del cfg.data.val.ins_ann_file
        del cfg.data.test.ins_ann_file
    except:
        pass
    
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    
    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = [1]

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # Build dataset
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False, #distributed
            shuffle=False)
    
    # Build the detector
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, weight_path, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(
        model, 
        data_loader, 
        False, 
        report_dir,
        show_score_thr
    )
    kwargs = {}
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in ['interval', 'tmpdir', 'start', 
        'gpu_collect', 'save_best', 'rule', 'dynamic_intervals']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric = ["bbox", "segm"], **kwargs))
    metric = dataset.evaluate(outputs, **eval_kwargs)
    _logger.info(metric)
    return model, outputs


def rsz_and_test_imgs(
        mmdet_cfg: str,
        weight_path: str,
        show_score_thr: float,
        mode: str,
        input_size: Optional[Tuple[int, int]] = None,
        report_dir: Optional[str] = None,
        background_img_path: str = None,
    ) -> None:
    cfg = Config.fromfile(mmdet_cfg)
    test_img_dir = cfg.data.test.img_prefix
    test_ann_file = cfg.data.test.ann_file
    if mode == "rsz_inference_imgs":
        model, results = rsz_inference_imgs(
            test_img_dir,
            mmdet_cfg,
            weight_path,
            input_size,
            show_score_thr,
            report_dir
        )
    elif mode == "rsz_paste_inference_imgs":
        model, results = rsz_paste_inference_imgs(
            test_img_dir,
            mmdet_cfg,
            weight_path,
            background_img_path,
            show_score_thr,
            report_dir
        )
    test_dataset = build_dataset(cfg.data.test)
    kwargs = {}
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in ['interval', 'tmpdir', 'start', 
        'gpu_collect', 'save_best', 'rule', 
        'dynamic_intervals']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric = ["bbox", "segm"], **kwargs))
    metric = test_dataset.evaluate(results, **eval_kwargs)
    print(metric)
    return model, results


def inference_imgs(
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


def rsz_inference_imgs(
        img_dir: str,
        mmdet_cfg: str,
        weight_path: str,
        input_size: Tuple[int, int],
        show_score_thr: float,
        report_dir: Optional[str] = None,
    ) -> EvaluateResultType:
    model = init_detector(mmdet_cfg, weight_path, device='cuda:0')
    time_deltas = []
    results = []
    i = 0
    for img_path, img, rsz_size in rsz_imgs_from_csv_info(img_dir, input_size):
        img = np.asarray(img).astype(np.uint8)
        t1 = time.time()
        result = inference_detector(model, img)
        t2 = time.time()
        time_deltas.append(t2 - t1)
        bboxes, seg_masks = result[0][0], result[1][0]
        org_bboxes = restore_bboxes(
            bboxes,
            rsz_size,
            input_size
        )
        valid_bbox_idx = remove_irregular_bboxes(
            org_bboxes,
            input_size,
            0.8
        )
        org_seg_masks = restore_seg_mask(
            seg_masks,
            rsz_size,
            input_size
        )
        valid_seg_mask_idx = remove_irregular_seg_masks(
            org_seg_masks,
            input_size,
            0.8
        )
        valid_idx = set(valid_bbox_idx).intersection(set(valid_seg_mask_idx))
        valid_org_bboxes = [org_bboxes[i] for i in valid_idx]
        valid_org_bboxes = np.concatenate(valid_org_bboxes, axis=0)
        valid_org_seg_masks = [org_seg_masks[i] for i in valid_idx]
        result[0][0] = valid_org_bboxes
        result[1][0] = valid_org_seg_masks
        if report_dir:
            try: 
                os.makedirs(Path(report_dir), exist_ok=True)
                img_name, img_ext = Path(img_path).name.split(".")
                visual_save_path = os.path.join(report_dir, f"{img_name}_pred.{img_ext}")
                model.show_result(
                    img_path, 
                    result, 
                    score_thr=show_score_thr, 
                    out_file=visual_save_path, 
                    font_size=9
                )
            except Exception as e:
                print(e)
                print(img_path)
                print(valid_bbox_idx)
                print(valid_seg_mask_idx)
        
        result_ = [([valid_org_bboxes], [valid_org_seg_masks])]
        encoded_result = [(bbox_result, encode_mask_results(mask_result))
            for bbox_result, mask_result in result_]
        results.extend(encoded_result)
        i += 1

    time_deltas = np.array(time_deltas)
    _logger.info(f"avg time consumed for inferencing \
                {len(time_deltas)} images is {time_deltas.mean()}")
    return model, results


def rsz_paste_inference_imgs(
        img_dir: str,
        mmdet_cfg: str,
        weight_path: str,
        background_img_path: str,
        show_score_thr: float,
        report_dir: Optional[str] = None,
    ) -> EvaluateResultType:
    back_img = pil_image.open(background_img_path).convert("RGB")
    input_size = (back_img.height, back_img.width)
    model = init_detector(mmdet_cfg, weight_path, device='cuda:0')
    time_deltas = []
    results = []
    i = 0
    for img_path, img, rsz_size in rsz_paste_imgs_from_csv_info(img_dir, back_img):
        img = np.asarray(img).astype(np.uint8)
        t1 = time.time()
        result = inference_detector(model, img)
        t2 = time.time()
        time_deltas.append(t2 - t1)
        bboxes, seg_masks = result[0][0], result[1][0]
        org_bboxes = restore_bboxes(
            bboxes,
            rsz_size,
            input_size
        )
        valid_bbox_idx = remove_irregular_bboxes(
            org_bboxes,
            input_size,
            0.8
        )
        org_seg_masks = restore_seg_mask(
            seg_masks,
            rsz_size,
            input_size
        )
        valid_seg_mask_idx = remove_irregular_seg_masks(
            org_seg_masks,
            input_size,
            0.8
        )
        valid_idx = set(valid_bbox_idx).intersection(set(valid_seg_mask_idx))
        valid_org_bboxes = [org_bboxes[i] for i in valid_idx]
        valid_org_bboxes = np.concatenate(valid_org_bboxes, axis=0)
        valid_org_seg_masks = [org_seg_masks[i] for i in valid_idx]
        result[0][0] = valid_org_bboxes
        result[1][0] = valid_org_seg_masks
        if report_dir:
            try: 
                os.makedirs(Path(report_dir), exist_ok=True)
                img_name, img_ext = Path(img_path).name.split(".")
                visual_save_path = os.path.join(report_dir, f"{img_name}_pred.{img_ext}")
                model.show_result(
                    img_path, 
                    result, 
                    score_thr=show_score_thr, 
                    out_file=visual_save_path, 
                    font_size=9
                )
            except Exception as e:
                print(e)
                print(img_path)
                print(valid_bbox_idx)
                print(valid_seg_mask_idx)
        
        result_ = [([valid_org_bboxes], [valid_org_seg_masks])]
        encoded_result = [(bbox_result, encode_mask_results(mask_result))
            for bbox_result, mask_result in result_]
        results.extend(encoded_result)
        i += 1

    time_deltas = np.array(time_deltas)
    _logger.info(f"avg time consumed for inferencing \
                {len(time_deltas)} images is {time_deltas.mean()}")
    return model, results