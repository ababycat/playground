import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

__all__ = ["SegData"]

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