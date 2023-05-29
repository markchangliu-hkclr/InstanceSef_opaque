# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Image transforms that are usually implemented on 
data before feeding them into models.
"""


import os
import torchvision.transforms.functional as F
from autostore.core.logging import get_logger
from PIL import Image as pil_image
from PIL.Image import Image
from typing import Generator, Any, Tuple


_logger = get_logger(__name__)


############### Image Transformations ###############

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
