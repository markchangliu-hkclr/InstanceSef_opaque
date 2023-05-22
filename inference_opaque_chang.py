from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import argparse
import cv2
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--choice', type=int, default=0, help='specify config index')
opt, _ = parser.parse_known_args()

show_score_thr = 0.8
folder = "/home/xjgao/Dataset/Autostore/Labeled_Autostore_all/test"
sub_folder_list = ['1']

if opt.choice==0:
    cfg_file = './configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
    model_dir = 'exps_opaque' + str(opt.choice) 
    checkpoint_name = 'epoch_2_small' #'epoch_2_small'
    print("used config file: {}".format('mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'))
elif opt.choice==1:
    cfg_file = './configs/mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ_opaque.py'
    model_dir = 'exps_opaque' + str(opt.choice) 
    checkpoint_name = 'epoch_2_small' #'epoch_2_small'
    print("used config file: {}".format('mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ_opaque.py'))
elif opt.choice==2:
    cfg_file = './configs/mask2former_r50_lsj_8x2_50e_coco_opaque.py'
    model_dir = 'exps_opaque' + str(opt.choice)
    checkpoint_name = 'iter_1500_small'
    print('used config file: {}'.format('mask2former_r50_lsj_8x2_50e_coco_opaque.py'))
elif opt.choice==3:
    cfg_file = './configs/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_opaque.py'
    model_dir = 'exps_opaque' + str(opt.choice)
    checkpoint_name = 'iter_1500_small'
    print('used config file: {}'.format('mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_opaque.py'))

checkpoint_file = model_dir + '/' + checkpoint_name + '.pth'
model = init_detector(cfg_file, checkpoint_file, device='cuda:0')
time_deltas = []

for id in sub_folder_list:
    img_dir = os.path.join(folder, id)

    for img_name in os.listdir(img_dir):
        if img_name.split(".")[-1]!="png" and img_name.split(".")[-1]!="jpg":
            continue

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        t1 = time.time()
        result = inference_detector(model, img)
        t2 = time.time()
        time_deltas.append(t2 - t1)

        outfile = os.path.join(model_dir, 'inference_result', f'{img_name}_inference.jpg')
        # model.show_result(img_path, result, score_thr=show_score_thr, out_file=outfile, font_size=9)

time_deltas = np.array(time_deltas)
print(f"avg time consumed for inferencing {len(time_deltas)} images is {time_deltas.mean()}s")