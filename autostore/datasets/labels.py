# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu


"""Label process module."""


import json
import labelme2coco
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image as pil_image
from pycocotools.coco import COCO


def cvt_labelme2coco(
        labelme_folder_path: str
    ) -> None:
    # for class-agnostic instance segmentation
    save_json_path = labelme_folder_path

    if os.path.isfile(os.path.join(save_json_path, 'dataset.json')):
        os.remove(os.path.join(save_json_path, 'dataset.json'))

    labelme2coco.convert(labelme_folder_path, save_json_path)

    SRC = save_json_path
    with open(os.path.join(SRC, '{}.json'.format('dataset')), 'r') as f:
        ori_json = json.load(f)

    for i in range(len(ori_json['annotations'])):
        print(ori_json['annotations'][i]["id"])
        ori_json['annotations'][i]['category_id'] = 0
    ori_json['categories'] = [{'id':0, 'name': 'object'}]

    with open(os.path.join(SRC, '{}.json'.format('dataset')), 'w') as f:
        json.dump(ori_json, f)


def visualize_coco(
        coco_ann_file: str
    ) -> None:
    pass