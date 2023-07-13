# Author: Chang Liu @ HKCLR


"""Utility funcs."""


import os
import shutil
from pathlib import Path
from typing import List

from autostore.core.io import load_img_paths


def train_val_split(
        data_dirs: List[str],
        output_dir: str,
        split_ratio: float = 0.7
    ) -> None:
    data = []
    for data_dir in data_dirs:
        img_generator = load_img_paths(data_dir)
        for img_path, _, _ in img_generator:
            img_path = Path(img_path)
            parent_path = img_path.parent
            img_name = img_path.stem
            json_path = os.path.join(parent_path, f"{img_name}.json")
            data.append((str(img_path), json_path))
    
    train_data = data[:int(len(data) * split_ratio)]
    val_data = data[int(len(data) * split_ratio):]

    train_folder_path = os.path.join(output_dir, "train")
    val_folder_path = os.path.join(output_dir, "val")
    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(val_folder_path, exist_ok=True)

    for img_path, json_path in train_data:
        new_img_path = os.path.join(train_folder_path, Path(img_path).name)
        new_json_path = os.path.join(train_folder_path, Path(json_path).name)
        shutil.copy(img_path, new_img_path)
        shutil.copy(json_path, new_json_path)

    for img_path, json_path in val_data:
        new_img_path = os.path.join(val_folder_path, Path(img_path).name)
        new_json_path = os.path.join(val_folder_path, Path(json_path).name)
        shutil.copy(img_path, new_img_path)
        shutil.copy(json_path, new_json_path)