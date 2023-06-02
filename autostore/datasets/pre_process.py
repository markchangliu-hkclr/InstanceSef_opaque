# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu


"""Pre process images and labels."""


import numpy as np
import os
import PIL.Image as pil_image
from autostore.datasets.transforms import (
    resize_pad_single_image,
    pad_single_img,
    paste_single_img,
    crop_single_img
)
from pathlib import Path
from PIL.Image import Image
from typing import List, Tuple, Generator


def get_rsz_info(
        rsz_file_path: str
    ) -> List[Tuple[str, int]]:
    rsz_info = []
    folder_path = Path(rsz_file_path).parent
    with open(rsz_file_path, 'r') as csv_file:
        for line in csv_file:
            line = line.strip()
            img_name, category, rsz_ratio = line.split(", ")
            img_path = os.path.join(folder_path, img_name)
            rsz_info.append((img_path, float(rsz_ratio)))
    return rsz_info


def rsz_imgs_from_csv_info(
        img_dir: str,
        input_size: Tuple[int, int]
    ) -> Generator[Tuple[str, Image, Tuple[int, int]], None, None]:
    rsz_file_path = os.path.join(img_dir, "resize.csv")
    rsz_info = get_rsz_info(rsz_file_path)
    for img_path, rsz_ratio in rsz_info:
        img = pil_image.open(img_path).convert("RGB")
        img, rsz_size = resize_pad_single_image(
            img,
            new_ratio = (rsz_ratio, rsz_ratio),
        )
        img = pad_single_img(img, input_size)
        # img = np.asarray(img)
        yield img_path, img, rsz_size


def rsz_paste_imgs_from_csv_info(
        img_dir: str,
        background_img: pil_image.Image
    ) -> Generator[Tuple[str, Image, Tuple[int, int]], None, None]:
    rsz_file_path = os.path.join(img_dir, "resize.csv")
    rsz_info = get_rsz_info(rsz_file_path)
    for img_path, rsz_ratio in rsz_info:
        img = pil_image.open(img_path).convert("RGB")
        img, rsz_size = resize_pad_single_image(
            img,
            new_ratio = (rsz_ratio, rsz_ratio),
        )
        img = crop_single_img(img, rsz_size)
        img = paste_single_img(img, background_img)
        yield img_path, img, rsz_size