import os
from pathlib import Path

import cv2
from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from transform import _ToTensor

   
class LaserDataset(Dataset):
    def __init__(self, datapath, class_nums, train=True, transform=None):
        """datapath should have two folders when train
        is True, and one folder when train=False.
        """
        self.file_path = datapath
        self.transform = transform
        self.class_nums = class_nums
        self.train = train
        if self.train:
            self.x_fns, self.y_fns = self.get_fns(self.file_path, train=True)
        else:
            self.x_fns = self.get_fns(self.file_path, train=False)
        img_test = cv2.imread(self.x_fns[0])
        self.init_grid(img_test.shape[0], img_test.shape[1])
        
    def __getitem__(self, index):
        if self.train:
            # x_fn, y_fn = self.x_fns[index], self.y_fns[index]
            x_fn, y_fn = self.x_fns[index], self.y_fns[index]
            x, y = self.get_data_pair(x_fn, y_fn)
            if self.transform is not None:
                x, y = self.transform(x, y)
            else:
                x, y = _ToTensor()(x), _ToTensor()(y)
            return x, y
        else:
            # fn = self.x_fns[index]
            fn = self.x_fns[index]
            x = Image.open(fn)
            if self.transform is not None:                
                x = self.transform(x)
            else:
                x = _ToTensor()(x)
            return x
        
    def __len__(self):
        return len(self.y_fns)

    def __repr__(self):
        fmt_str = 'line laser dataset for segmentation'
        return fm_str

    def get_data_pair(self, x_fn, y_fn):
        x = Image.open(x_fn)
        y = self.get_label(y_fn)
        return x, y

    def get_label(self, filepath, is_one_hot=False):
        label = cv2.imread(filepath)[:, :, 2]
        if is_one_hot:
            one_hot = np.zeros((self.class_nums, label.shape[0], label.shape[1]), dtype=np.uint8)        
            one_hot[label.reshape(-1), self._hh, self._ww] = 255
            return one_hot
        else:
            label[label==1] = 255
            return label.reshape(1, label.shape[0], label.shape[1])

    def init_grid(self, shape0, shape1):
        h = np.arange(0, shape0)
        w = np.arange(0, shape1)
        ww, hh = np.meshgrid(w, h)

        self._ww = ww.reshape(-1)
        self._hh = hh.reshape(-1)

    def get_fns(self, filepath, train=True):
        filepath = Path(filepath)
        if train:
            x_path = Path('data')
            y_path = Path('label')
            x_fns = os.listdir(str(filepath/x_path))
            y_fns = os.listdir(str(filepath/y_path))
            x_fns = [str(filepath/x_path/Path(t)) for t in x_fns]
            y_fns = [str(filepath/y_path/Path(t)) for t in y_fns]
            return x_fns, y_fns
        else:
            x_path = Path('data')
            x_fns = os.listdir(str(filepath/x_path))
            x_fns = [str(filepath/x_path/Path(t)) for t in x_fns]
            return x_fns 


class SegData(Dataset):
    def __init__(self, dataset, train=True, transform=None):
        self.transform = transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels = dataset
        else:
            self.test_data, self.test_labels = dataset

    def __getitem__(self, index):
        """return img is a PIL.Image"""
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        
        img, target = Image.fromarray(img), target.transpose(2, 0, 1)
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'undersea segmentation'
        return fmt_str

class Seg_Data_Beta(LaserDataset):
    def __init__(self, datapath, class_nums, train=True, transform=None):
        super().__init__(datapath, class_nums, train=True, transform=None)
    
    def __repr__(self):
        fmt_str = 'undersea segmentation'
        return fmt_str