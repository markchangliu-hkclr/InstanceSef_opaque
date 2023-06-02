# Corporation: HKCLR 
# Project: AutoStore picking robots
# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Some I/O operations that could be helpful 
in other modules.
"""

import os
import shutil
from pathlib import Path
from autostore.core.logging import get_logger
from typing import Generator, Tuple, Any, Dict


_logger = get_logger(__name__)


############### File I/O Operations ###############

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
    for root, subdirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                img_path = Path(os.path.join(root, file))
                img_name = img_path.stem
                img_ext = img_path.suffix
                yield str(img_path), img_name, img_ext


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


def org_path_commodity_type_path_map(
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
        >>>     "toothpaste": ((0, 6)),
        >>>     "tea_box": ((7, 10)),
        >>>    "napkin_box": ((11, 15)),
        >>>    "bestbuy_can": ((16, 20)),
        >>>    "coke_can": ((21, 25)),
        >>>    "drink_bottle": ((26, 39), (45, 53)),
        >>>    "wafer_biscuit": ((40, 44)),
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
    for commodity, indice in commodity_img_index.items():
        for start_idx, end_idx in indice:
            new_img_dir = os.path.join(curr_path, "commodity_group", commodity)
            img_indice = (str(i).rjust(5, "0") for i in range(start_idx, end_idx + 1))
            img_names = (f"color_{img_idx}.png" for img_idx in img_indice)
            for img_name in img_names:
                img_path = os.path.join(img_dir, img_name)
                new_img_path = os.path.join(new_img_dir, img_name)
                yield img_path, new_img_path


def org_path_new_path_unchanged_rel_map(
        root_dir: str,
        folder_dir: str,
        new_root_dir: str,
    ) -> Generator[Tuple[str, str], None, None]:
    """
    Generate a list of (org_path, new_path) mappings, where oth_path
    and new_path have the same relative path relative to root_dir and 
    new_root_dir. In other words, `new_path.relative_to(new_root_dir)
    == org_path.relative_to(root_dir)`.

    Args:
        root_dir (str): 
            The root path adopted to compute the relative path of org_path.
        folder_dir (str):
            The folder path that files are stored at. `rel_path = file_path.
            relative_to(root_dir)`
        new_root_dir (str):
            the new_path of files equals `new_root_dir / rel_path`.
    """
    new_root_dir = Path(new_root_dir)
    for root, subdirs, files in os.walk(folder_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                old_path = Path(os.path.join(root, file))
                rel_path = old_path.relative_to(root_dir)
                new_path = new_root_dir / rel_path
                yield str(old_path), str(new_path)