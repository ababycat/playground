import os
from pathlib import Path
import cv2

import numpy as np
from PIL import Image

__all__ = ["get_label", "get_data_pair", "get_data"]

def get_label(raw_label, class_nums):
    label = np.zeros((raw_label.shape[0], raw_label.shape[1], class_nums), dtype=np.uint8)
    for i in range(class_nums):
        label[raw_label==i, i] = 255
    # r, g, b = cv2.split(label)
    # label = np.dstack((r, b, g))
    # label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
    return label

def get_data_pair(file_path_pair, class_nums):
    img_path, label_path = file_path_pair
    img = np.array(Image.open(img_path))
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    raw_label = np.array(Image.open(label_path))[:, :, 0]
    label = get_label(raw_label, class_nums)
    return img, label

def get_data(root, class_nums):
    dataset_path = root
    label_dir_name = 'label'
    data_dir_name = 'data'

    fns = os.listdir(str(dataset_path/label_dir_name))

    data = []
    labels = []
    for idx, fn in enumerate(fns):
        label_path =  dataset_path/label_dir_name/fn
        data_path =  dataset_path/data_dir_name/fn
        img, label = get_data_pair((data_path, label_path), class_nums)
        data.append(img)
        labels.append(label)
    return (data, labels)


