# *coding:utf-8 *

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import random

from lib.utils.image import randomHueSaturationValue
from lib.utils.image import randomShiftScaleRotate
from lib.utils.image import randomHorizontalFlip
from lib.utils.image import randomVerticleFlip, randomRotate90


class BinarySegDataset(data.Dataset):
    def image_transform(self):
        img_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=1),transforms.ColorJitter(brightness=0.2)])
        mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        return img_transform, mask_transform
    
    def __getitem__(self, index):
        img = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)[4:]
        img = img[:800]
        img = cv2.resize(img, (self.opt.height, self.opt.width))
        # cv2.imshow("",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        mask = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)[4:]
        mask = mask[:800]
        mask = cv2.resize(mask, (self.opt.height, self.opt.width))
        transform = self.image_transform()
        # Data augmentation
        if self.opt.color_aug:
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-30, 30),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

        if self.opt.shift_scale:
            img, mask = randomShiftScaleRotate(img, mask,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.1, 0.1),
                                               aspect_limit=(-0.1, 0.1),
                                               rotate_limit=(-0, 0))

        if self.opt.HorizontalFlip:
            img, mask = randomHorizontalFlip(img, mask)

        if self.opt.VerticleFlip:
            img, mask = randomVerticleFlip(img, mask)

        if self.opt.rotate_90:
            img, mask = randomRotate90(img, mask)
        
        p = 0.5
        random_number = random.random()
        if random_number < p:
            img_pil = Image.fromarray(img.astype('uint8'))
            mask_pil = Image.fromarray(mask.astype('uint8'))
            img = transform[0](img_pil)
            mask = transform[1](mask_pil)
            img = np.array(img)
            mask = np.array(mask)
        
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32)
        img = img[np.newaxis, :, :]  # 添加通道维度，变成 (1, height, width)
        img = img / 255.0 * 3.2 - 1.6
        mask = np.array(mask, np.float32)
        mask = mask[np.newaxis, :, :]  # 添加通道维度，变成 (1, height, width)
        mask = mask/ 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        ret = {'input': img, 'gt': mask}
        return ret
