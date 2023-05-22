# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu
# Date last modified: May 19, 2023
# Some codes are referenced from https://github.com/facebookresearch/pycls


"""Preprocess operations that are usually implemented on 
data before feeding them into models.
"""


import os
import numpy as np
import shutil
import torchvision.transforms.functional as F
from core.logging import get_logger
from pathlib import Path
from PIL import Image as pil_image
from PIL.Image import Image
from typing import Generator, Any, List, Union, Tuple, Dict


_logger = get_logger(__name__)


############### File I/O Operations ###############

def cp_files(
        file_paths: Generator[Tuple[str, str], Any, Any]
    ) -> None:
    """Copy a list of files from their src paths to dst paths.

    This function automatically create the parent folders of dst paths 
    if they do not exist.

    Arg:
        file_paths (Generator[Tuple[str, str], Any, Any]): 
            A generator that yield a tuple (src_path, dst_path) for each
            file that waits to be copied.

    Returns:
        None
    """
    for src_path, dst_path in file_paths:
        dst_parent_dir = Path(dst_path).parent
        os.makedirs(dst_parent_dir, exist_ok=True)
        shutil.copyfile(src_path, dst_path)


def find_img_paths_for_commodity_types(
        commodity_img_index: Dict[str, Tuple[int, int]],
        img_dir: str
    ) -> Generator[Tuple[str, str], None, None]:
    """Given that you have images of different commodity types 
    saved under a folder, and a dictionary mapping commodity types 
    to image indices. You now want to place those images into 
    folders named as commodity types. This function can generate 
    a tuple (src_path, dst_path) for each image that allows 
    you to implement copy operations.
    
    This function assumes that the naming format for images 
    is "color_{img_idx}.png".

    The dst paths locate under "CURRENT_WORK_DIR/commodity_group".

    Args:
        commodity_img_index (Dict[str, Tuple[int, int]]): 
            A dictionary that maps commodity types to image indice.
        img_dir (str):
            The src folder dir of images.

    Yield:
        img_paths (Generator[Tuple[str, str], None, None]):
            A tuple (src_path, dst_path) for helping subsequent 
            operations, such as "cp" or "mv".
    
    Examples:
        >>> _COMMODITY_IMG_INDEX = {
        >>>     "toothpaste": (0, 6),
        >>>     "tea_box": (7, 10),
        >>>    "napkin_box": (11, 15),
        >>>    "bestbuy_can": (16, 20),
        >>>    "coke_can": (21, 25),
        >>>    "drink_bottle": (26, 39),
        >>>    "wafer_biscuit": (40, 44),
        >>>    "drink_bottle": (45, 53)
        >>> }
        >>> 
        >>> img_dir = "/home/whoever/data"
        >>> 
        >>> img_paths = find_img_paths_for_commodity_types(
        >>>     _COMMODITY_IMG_INDEX,
        >>>     img_dir
        >>> )
        >>> 
        >>> next(img_paths)
        (/home/whoever/data/color_00000.png, 
        CURRENT_WORK_DIR/commodity_group/toothpaste/color_00000.png)
    """
    curr_path = os.getcwd()
    for commodity, (start_idx, end_idx) in commodity_img_index.items():
        new_img_dir = os.path.join(curr_path, "commodity_group", commodity)
        img_indice = (str(i).rjust(5, "0") for i in range(start_idx, end_idx + 1))
        img_names = (f"color_{img_idx}.png" for img_idx in img_indice)
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            new_img_path = os.path.join(new_img_dir, img_name)
            yield img_path, new_img_path


############### Image Transformations ###############

def load_img_paths(img_dir: str) -> Generator[str, None, None]:
    """Load the paths of all images under a directory and 
    its child directories.
    
    This function regards files with one of the extensions 
    ".jpg", ".jpeg", ".png" as images.

    This function does not check if an image file is valid
    or broken.

    Args:
        img_dir (str):
            The ditectory that you want to search at.
    
    Yield:
        img_paths (Generator[str, None, None]):
            A generator that yield img paths.
    """
    for item in os.listdir(img_dir):
        if item.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(img_dir, item)
            img_name, img_ext = item.split(".")
            yield img_path, img_name, img_ext


def resize_pad_single_image(
        img: Image,
        new_size: Tuple[int, int]
    ) -> Image:
    """Resize (adn padded) an image to `new_size` with unchanged 
    aspect ratio. The dimension, either the height or the width, 
    that is smaller than `new_size` after resizing will be zero-paded.

    Args:
        img (pil_image.Image):
            A PIL Image instance.
        new_size (Tuple[int, int]):
            The output (h, w) of the image.
    
    Returns:
        new_img (pil_image.Image):
            The resized (and padded) image.
    """
    h, w = img.height, img.width
    new_h, new_w = new_size
    rsz_scale = min(new_h / h, new_w / w)
    rsz_h, rsz_w = int(rsz_scale * h), int(rsz_scale * w)
    img = F.resize(img, (rsz_h, rsz_w))
    pad_left = (new_w - rsz_w) // 2
    pad_right = new_w - pad_left - rsz_w
    pad_top = (new_h - rsz_h) // 2
    pad_bottom = new_h - pad_top - rsz_h
    img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom))
    return img


# def resize_single_img_by_ratio(
#         img: Image,
#         new_ratio: Tuple[int, int]
#     ) -> Image:
#     h, w = img.height, img.width
#     new_h, new_w = int(h * new_ratio[0]), int(w * new_ratio[1])
#     return 


def resize_pad_imgs(
        img_paths: Generator[str, Any, Any],
        new_ratio: Tuple[float, float],
        new_size: Tuple[int, int],
    ) -> Generator[Image, None, None]:
    """Apply `resize_pad_single_img` on a list of images.
    The new size of images can be specified by either `rsz_ratio`
    or `rsz_size`. 

    You are supposed to provide one of `rsz_ratio` and `rsz_size`,
    otherwise an error will arise. If you provide both, the function 
    will use `rsz_ratio`.
    
    Args:
        img_paths (Generator[str, Any, Any]):
            A generator that can output image paths.
        new_ratio: Tuple[float, float]:
            The ratio of (new_h, new_w) to (current_h, current_w).
        new_size: Tuple[int, int]:
            The pixel size of (new_h, new_w).
    
    Yield:
        new_imgs (Generator[pil_image.Image, None, None]):
            A generator producing resized images.
    """
    assert new_ratio or new_size, \
        "You must provide one of 'rsz_ratio' and 'rsz_size'."
    for img_path in img_paths:
        try:
            img = pil_image.open(img_path)
        except Exception as e:
            raise e
        h, w = img.height, img.width
        if new_ratio:
            new_size_ = (int(new_ratio[0] * h), int(new_ratio[1] * w))
        else:
            new_size_ = new_size
        img = resize_pad_single_image(img, new_size_)
        yield img


def pad_single_img(
        img: Image,
        new_size: Tuple[int, int]
    ) -> Image:
    """Zero-pad an image to a new size.
    
    Args:
        img (pil_image.Image):
            An PIL Image.
        new_size (Tuple[int, int]):
            The pixel size of (new_h, new_w).
    
    Returns:
        new_img:
            The zero-paded image.
    """
    h, w = img.height, img.width
    new_h, new_w = new_size
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom))
    return img


if __name__ == "__main__":
    pass
