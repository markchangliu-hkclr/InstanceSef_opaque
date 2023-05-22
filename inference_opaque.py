from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import os.path as osp
import time
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt


config_file = './configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
# config_file = './configs/mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ_opaque.py'
# config_file = './configs/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_opaque.py'
# config_file = './configs/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
time_start = time.time()
checkpoint_name = 'epoch_2_small'
# checkpoint_name = 'epoch_2_2303241551_LSJ'
checkpoint_file  = 'exps_opaque0/' + checkpoint_name +'.pth'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("Time consumed model load: ", time.time()-time_start)

# x_min, x_max, y_min, y_max = 430, 1000, 170, 560
# x_min, x_max, y_min, y_max = 460, 930, 190, 530
# x_min, x_max, y_min, y_max = 410, 1010, 110, 560
# x_min, x_max, y_min, y_max = 590, 890, 150, 380 # photoneo autostore
x_min, x_max, y_min, y_max = 395, 845, 175, 525 # photoneo autostore v2
# x_min, x_max, y_min, y_max = 560, 910, 80, 340 # AutoStore black bin
# x_min, x_max, y_min, y_max = 545, 955, 75, 345 # photoneo poc
# x_min, x_max, y_min, y_max = 0, 1280, 0, 720 # cropped images
# x_min, x_max, y_min, y_max = 580, 905, 85, 325 # Test case

# test a single image and show the results
# folder = '/home/xjgao/Dataset/Autostore/Labeled_Autostore_all/test'
# folder = '/home/xjgao/Dataset/Autostore/Autostore_230217'
# folder = '/home/xjgao/Dataset/Autostore/Bin_det'
# folder = '/home/xjgao/Dataset/Autostore/Failure_case'
folder = "/home/xjgao/Dataset/Autostore/Labeled_Autostore_all/test"


# sub_folder_list = os.listdir(folder)
sub_folder_list = ['1']

for id in sub_folder_list:
    img_dir = os.path.join(folder, id)

    for img_name in os.listdir(img_dir):
        if img_name.split(".")[-1]!="png" and img_name.split(".")[-1]!="jpg":
            continue

        img_path = osp.join(img_dir, img_name)
        print(img_path)
        img = cv2.imread(img_path)

        img_height, img_weight = img.shape[0], img.shape[1]
        cropped_image = img[y_min:y_max, x_min:x_max]
        time_start = time.time()
        result = inference_detector(model, cropped_image)
        print("Time consumed: ", time.time()-time_start)
        # model.show_result(cropped_image, result, out_file=osp.join(img_dir, 'seg_results', img_name.split('.')[0]+'_crop_result.png'))
        model.show_result(cropped_image, result, out_file=osp.join('./tmp', img_name.split('.')[0]+'_crop_result.png'))
        bboxes_, masks_ = result
        masks_ = np.asarray(masks_[0])
        masks = np.zeros((masks_.shape[0], img_height, img_weight))
        if len(masks)==0: continue
        masks[:, y_min:y_max, x_min:x_max] = masks_
        np.savez_compressed(os.path.join(img_dir, 'masks_' + img_name.split('.')[0]), masks)

        ## save cropped image
        # plt.imsave(os.path.join(img_dir, 'seg_results', img_name.split('.')[0]+'_crop.png'), img[y_min:y_max, x_min:x_max, ::-1])
        # plt.imsave(os.path.join('./tmp', img_name.split('.')[0]+'_crop.png'), img[y_min:y_max, x_min:x_max, ::-1])
