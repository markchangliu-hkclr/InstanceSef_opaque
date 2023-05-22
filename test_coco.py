import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.apis import multi_gpu_test, single_gpu_test
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--choice', type=int, default=0, help='specify config index')
opt, _ = parser.parse_known_args()


work_dir = 'result_opaque'
model_dir = 'exps_opaque'

if opt.choice==0:
    cfg = Config.fromfile('/home/xjgao/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py')
    checkpoint_name = 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00'
    print("used config file: {}".format('mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py'))
elif opt.choice==1:
    cfg = Config.fromfile('old/mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ.py')
    checkpoint_name = 'model_final_14d201-mmdet'
    print("used config file: {}".format('mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ.py'))
elif opt.choice==2:
    cfg = Config.fromfile('old/mask2former_r50_lsj_8x2_50e_coco.py')
    checkpoint_name = 'mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b'
    print("used config file: {}".format('mask2former_r50_lsj_8x2_50e_coco.py'))
elif opt.choice==3:
    cfg = Config.fromfile('old/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco.py')
    checkpoint_name = 'mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_20220508_091649-4a943037'
    print("used config file: {}".format('mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco.py'))    


checkpoint =  osp.join('/home/xjgao/InstanceSeg/checkpoints', checkpoint_name + '.pth')


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

# Modify dataset type and path
cfg.data.test.img_prefix = '/data/xwchi/val2017'
cfg.data.test.ann_file = osp.join('/data/xwchi/annotations/instances_val2017.json')

# cfg.data.test.img_prefix = '/home/xjgao/Dataset/Autostore/Labeled_Autostore_all/test/1'
# cfg.data.test.ann_file = osp.join('/home/xjgao/Dataset/Autostore/Labeled_Autostore_all/test/1/dataset.json')

eval = ['bbox', 'segm']
out = None #'out.pkl'
show_dir = work_dir
show_score_thr = 0.05

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = [0]

# We can initialize the logger for training and have a look
# at the final config
# print(f'Config:\n{cfg.pretty_text}')

# in case the test dataset is concatenated
cfg.data.test.samples_per_gpu = 2
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
checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=cfg.gpu_ids)
# outputs = single_gpu_test(model, data_loader, False, show_dir,
#                                   show_score_thr)
outputs = single_gpu_test(model, data_loader, False)
if out:
    print(f'\nwriting results to {out}')
    mmcv.dump(outputs, out)


kwargs = {}
eval_kwargs = cfg.get('evaluation', {}).copy()
# hard-code way to remove EvalHook args
for key in [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule', 'dynamic_intervals'
            ]:
    eval_kwargs.pop(key, None)
eval_kwargs.update(dict(metric=eval, **kwargs))
if 'ins_results' in outputs[0]:
    outputs = [result['ins_results'] for result in outputs]
metric = dataset.evaluate(outputs, **eval_kwargs)
print(metric)
# metric_dict = dict(config=config_file, metric=metric)
# json_file = work_dir + '/' + checkpoint_name+'eval.json'
# if work_dir is not None:
#     mmcv.dump(metric_dict, json_file)
