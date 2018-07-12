import types
import random
import math

import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

__all__ = ["_Compose", "_ColorJitter", "_Resize", "_RandomApply", "_RandomRotation",
            "_Lambda", "_RandomApply", "_RandomGrayscale", "_RandomCrop", 
            "_RandomHorizontalFlip", "_RandomVerticalFlip", "_ToTensor", "_Lambda"]

class _Compose(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)
       
    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class _ColorJitter(transforms.ColorJitter):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(_Lambda(lambda img, target: (F.adjust_brightness(img, brightness_factor), target_op(target, F.adjust_brightness, brightness_factor))))
        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(_Lambda(lambda img, target: (F.adjust_contrast(img, contrast_factor), target_op(target, F.adjust_contrast, contrast_factor))))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(_Lambda(lambda img, target: (F.adjust_saturation(img, saturation_factor), target_op(target, F.adjust_saturation, saturation_factor))))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(_Lambda(lambda img, target: (F.adjust_hue(img, hue_factor), target_op(target, F.adjust_hue, hue_factor))))

        random.shuffle(transforms)
        transform = _Compose(transforms)

        return transform

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Input image.
            target (numpy array CxHxW): Input image
        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img, target)
    
class _Resize(transforms.Resize):
    """input: PIL Image
    """
    def __init__(self, scale_range=(0.5, 1.5), raw=1080, interpolation=Image.BILINEAR):
        scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
        s = int(scale*raw)
        size = (s, s)
        super().__init__(size, interpolation)
        
    def __call__(self, img, target):
        return F.resize(img, self.size, self.interpolation), target_op(target, F.resize, self.size, self.interpolation)
    
class _RandomApply(transforms.RandomApply):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p)

    def __call__(self, img, target):
        if self.p < random.random():
            return img, target
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
    
class _RandomGrayscale(transforms.RandomGrayscale):
    def __init__(self, p=0.1):
        super().__init__(p)

    def __call__(self, img, target):
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, 3), target_op(target, F.to_grayscale, 1)
        return img, target

class _RandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        super().__init__(size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)
        
    def __call__(self, img, target):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            target = target_op(target, F.pad, self.padding, self.fill, self.padding_mode)
            
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0), self.fill, self.padding_mode)
            target = target_op(target, F.pad, (int((1 + self.size[1] - target.size[0]) / 2), 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)), self.fill, self.padding_mode)
            target = target_op(target, F.pad, (0, int((1 + self.size[0] - target.size[1]) / 2)), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), target_op(target, F.crop, i, j, h, w)

class _RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img, target):
        if random.random() < self.p:
            return F.hflip(img), target_op(target, F.hflip)
        return img, target

class _RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img, target):
        if random.random() < self.p:
            return F.vflip(img), target_op(target, F.vflip)
        return img, target
    
class _RandomRotation(transforms.RandomRotation):
    def __init__(self, p, degrees, output_size=512, fill=0, padding_mode='symmetric', resample=Image.NEAREST, expand=False, center=None):
        super().__init__(degrees, resample=resample, expand=expand, center=center)
        self.p = p
        self.padding_mode = padding_mode
        self.fill = fill
        self.size = output_size
        
    def __call__(self, img, target):
        if random.random() < self.p:
            self.padding = int(math.ceil((1.42-1)*img.size[0]*0.5))
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            target = target_op(target, F.pad, self.padding, self.fill, self.padding_mode)

            angle = self.get_params(self.degrees)
            img, target = F.rotate(img, angle, self.resample, self.expand, self.center), target_op(target, F.rotate, angle, self.resample, self.expand, self.center)
            return F.center_crop(img, self.size), target_op(target, F.center_crop, self.size)
        else:
            return img, target
        
def CxHxW2HxWxC(func):
    def wrapper(value):
        if isinstance(value, np.ndarray):
            value_t = value.transpose(1, 2, 0)
        else:
            value_t = value
        return func(value_t)
    return wrapper

@CxHxW2HxWxC
def _toTensor(pic):
    return F.to_tensor(pic)

class _ToTensor(transforms.ToTensor):
    
    def __call__(self, pic):
#         if isinstance(pic, np.ndarray):
#             pic = pic.transpose(1, 2, 0)
        return _toTensor(pic)

class _Lambda(transforms.Lambda):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, target):
        return self.lambd(img, target)
    
def target_op(target, op, *params):
    """target : a numpy array, (C*H*W)
        op : a function in torchvision.functional
        output : a numpy array"""
    out = []
    for c in range(target.shape[0]):
        p_img = Image.fromarray(target[c, :, :])
        out.append(np.array(op(p_img, *params)))
    out = tuple(out)
    return np.stack(out, axis=0)