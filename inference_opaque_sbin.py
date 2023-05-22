from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import os.path as osp
import time
import numpy as np
import cv2
import open3d as o3d
from utils import *

realsense_intrinsics = np.identity(3)
realsense_intrinsics[0, 0] = 909.052490234375 #898.684
realsense_intrinsics[1, 1] = 908.5682373046875 #898.684
realsense_intrinsics[0, 2] = 647.178955078125 #643.735
realsense_intrinsics[1, 2] = 337.7300109863281 #359.027
photoneo_to_realsense_mat =  np.array([[0.983396,  0.00351838,    0.181437,   -0.304045],
                        [0.000993297,    0.999693,  -0.0247696,   0.0225611],
                        [-0.181468,   0.0245385,    0.983091,   0.0915611],
                        [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])


# config_file = './configs/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
config_file = './configs/mask_rcnn_r50_caffe_fpn_detectron2_400ep_LSJ_opaque.py'
# config_file = './configs/cascade_mask_rcnn_r50_caffe_fpn_mstrain_3x_coco_opaque.py'
# config_file = './configs/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco_opaque.py'
time_start = time.time()
# checkpoint_name = 'epoch_2_12231502_small'
checkpoint_name = 'epoch_2_2303241551_LSJ'
checkpoint_file  = 'exps_opaque/' + checkpoint_name +'.pth'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("Time consumed model load: ", time.time()-time_start)

# test a single image and show the results
# folder = '/home/xjgao/Dataset/Autostore/Labeled_Autostore_all/test'
# folder = '/home/xjgao/Dataset/Autostore/Autostore_230217'
folder = '/home/xjgao/Dataset/Autostore/Autostore_230418_sbin'
# folder = '/home/xjgao/Dataset/Autostore/Failure_case'

sub_folder_list = os.listdir(folder)
# sub_folder_list = ['29']

for id in sub_folder_list:
    img_dir = os.path.join(folder, id)

    for img_name in os.listdir(img_dir):
        if img_name.split(".")[-1]!="png" and img_name.split(".")[-1]!="jpg":
            continue

        img_path = osp.join(img_dir, img_name)
        print(img_path)
        idx = img_name.split('.')[0].split('img_')[-1]
        img = cv2.imread(img_path)
        pcd = o3d.io.read_point_cloud(os.path.join(img_dir, 'ply_' + idx + '.ply'))

        scene_pc_array = np.asarray(pcd.points)
        multi_camera_registration = MultiCameraRegistration(realsense_intrinsics, photoneo_to_realsense_mat)

        # corners3D, bin_center_pose, bin_inner_width = locate_box_corners(scene_pc_array, bottom2camera = 1.08, top2camera = 1.05,
        #                                             box_shape = [0.28, 0.38], height_dir = [0.12187793, 0.00411619, 0.99253656], 
        #                                             bin_scale = 0.8, thickness = 0.018, show_result = False)
        # bin_h_2d, bin_w_2d = 380, 280

        corners3D, bin_center_pose, bin_inner_width = locate_box_corners(scene_pc_array, bottom2camera = 1.16, top2camera = 1.13,
                                                    box_shape = [0.28, 0.38], height_dir = [0.12187793, 0.00411619, 0.99253656], 
                                                    bin_scale = 0.8, thickness = 0.018, show_result = False)
        bin_h_2d, bin_w_2d = 355, 255
        
        bin_center_3d = np.mean(corners3D, axis=0)
        multi_camera_registration.calculate_project_index(np.array(corners3D + [bin_center_3d]))
        bin_corner_2d_x, bin_corner_2d_y = multi_camera_registration.get_project_index()
        bin_center_2d_x, bin_center_2d_y = int(bin_corner_2d_x[-1]), int(bin_corner_2d_y[-1])
        nextshortcorner_id = np.argmin(np.sum((corners3D[1:]-corners3D[0])**2, axis=1)) + 1
        rotated_angle = np.arctan((bin_corner_2d_x[0] - bin_corner_2d_x[nextshortcorner_id]) / (bin_corner_2d_y[0] - bin_corner_2d_y[nextshortcorner_id] + 1e-10))
        rotated_angle = -np.rad2deg(rotated_angle)
        M = cv2.getRotationMatrix2D((bin_center_2d_x, bin_center_2d_y), rotated_angle, 1)
        inv_M = cv2.getRotationMatrix2D((bin_center_2d_x, bin_center_2d_y), -rotated_angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        x_min, x_max, y_min, y_max = [bin_center_2d_x-bin_h_2d//2, bin_center_2d_x+bin_h_2d//2, bin_center_2d_y-bin_w_2d//2, bin_center_2d_y+bin_w_2d//2]

        img_height, img_width = img.shape[0], img.shape[1]
        cropped_image = img[y_min:y_max, x_min:x_max]
        time_start = time.time()

        ## save cropped images
        # os.makedirs(os.path.join(img_dir, 'crop'), exist_ok=True)
        # plt.imsave(os.path.join(img_dir, 'crop', 'img_' + idx +'.png'), img[y_min:y_max, x_min:x_max, ::-1])

        ## generate seg results
        result = inference_detector(model, cropped_image)
        print("Time consumed: ", time.time()-time_start)
        model.show_result(cropped_image, result, out_file=osp.join(img_dir, 'seg_results', img_name.split('.')[0]+'_crop_result.png'))
        bboxes_, masks_ = result
        masks_ = np.asarray(masks_[0])
        masks = np.zeros((masks_.shape[0], img_height, img_width))
        if len(masks)==0: continue
        masks[:, y_min:y_max, x_min:x_max] = masks_
        masks = np.transpose(masks, (1, 2, 0))
        masks = cv2.warpAffine(masks, inv_M, (img_width, img_height))
        masks = np.reshape(masks, (masks.shape[0], masks.shape[1], -1))
        masks = np.transpose(masks, (2, 0, 1))
        np.savez_compressed(os.path.join(img_dir, 'masks_' + img_name.split('.')[0]), masks)
