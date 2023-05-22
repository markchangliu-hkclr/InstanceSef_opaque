import mmcv
# from mmengine.config import Config
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp
import torch
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--choice', type=int, default=2, help='specify config index')
opt, _ = parser.parse_known_args()

if opt.choice==0:
    cfg = Config.fromfile('./configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py')
    cfg.load_from = '/home/xjgao/InstanceSeg/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    print("used config file: {}".format('mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque'))
elif opt.choice==1:
    cfg = Config.fromfile('./configs/mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ_opaque.py')
    cfg.load_from = '/home/xjgao/InstanceSeg/checkpoints/model_final_14d201-mmdet.pth'
    print("used config file: {}".format('mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ_opaque'))
elif opt.choice==2:
    cfg = Config.fromfile('./configs/mask2former_r50_lsj_8x2_50e_coco_opaque.py')
    del cfg.data.val.ins_ann_file
    del cfg.data.test.ins_ann_file
    cfg.load_from = '/home/xjgao/InstanceSeg/checkpoints/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'
    print("used config file: {}".format('mask2former_r50_lsj_8x2_50e_coco_opaque'))
elif opt.choice==3:
    cfg = Config.fromfile('./configs/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_opaque.py')
    cfg.load_from = '/home/xjgao/InstanceSeg/checkpoints/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_20220508_091649-4a943037.pth'
    print("used config file: {}".format('mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_opaque'))
# cfg = Config.fromfile('./configs/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_opaque.py')
# cfg.load_from = '/home/xjgao/InstanceSeg/checkpoints/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210707_002651-6e29b3a6.pth'
# cfg = Config.fromfile('./configs/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco_opaque.py')
# cfg.load_from = '/home/xjgao/InstanceSeg/checkpoints/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco_20210526_132339-3c33ce02.pth'


# Set up working dir to save files and logs.
cfg.work_dir = './exps_opaque' + str(opt.choice)
cfg.device = 'cuda'

# cfg.optimizer.lr = 5e-4
# cfg.lr_config.warmup = None
# cfg.lr_config.step = [25, 48]
# cfg.log_config.interval = 200

# cfg.runner.max_epochs = 2
# We can set the evaluation interval to reduce the evaluation times
# cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
# cfg.checkpoint_config.interval = 2

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = [0]

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)

try:
    ckpt_name = '/epoch_'+str(cfg.runner.max_epochs)
    checkpoint = torch.load(cfg.work_dir+ckpt_name+'.pth')
except:
    ckpt_name = '/iter_'+str(cfg.runner.max_iters)
    checkpoint = torch.load(cfg.work_dir+ckpt_name+'.pth')
os.remove(cfg.work_dir+ckpt_name+'.pth')
meta = checkpoint['meta']
checkpoint['optimizer'] = None
weights = checkpoint['state_dict']
state_dict = {"state_dict": weights, "meta": meta}
torch.save(state_dict, cfg.work_dir+ckpt_name+'_small.pth')
