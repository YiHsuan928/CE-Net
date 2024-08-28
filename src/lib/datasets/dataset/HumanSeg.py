# *coding:utf-8 *

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os

import torch.utils.data as data


class HumanSeg(data.Dataset):
    num_classes = 1
    default_resolution = [256, 256]
    mean = np.array([46.3758],
                    dtype=np.float32)#.reshape(1, 1, 3)
    std = np.array([38.8791],
                   dtype=np.float32)#.reshape(1, 1, 3)
    
    def __init__(self, opt, split):
        self.images = []
        self.labels = []
        self.opt = opt
        self.split = split

        if split == 'train':
            base_dir = os.path.join(opt.data_dir, 'train')
        elif split == 'valid':
            base_dir = os.path.join(opt.data_dir, 'valid')
        else:
            base_dir = os.path.join(opt.data_dir, 'test')

        self._load_data(base_dir)

        if len(self.images) == 0:
            raise ValueError(f"No images found in the dataset. Check the dataset directory: {base_dir}")

    def _load_data(self, base_dir):
        for subdir1 in os.listdir(base_dir):
            subdir1_path = os.path.join(base_dir, subdir1)
            if os.path.isdir(subdir1_path):
                for subdir2 in os.listdir(subdir1_path):
                    subdir2_path = os.path.join(subdir1_path, subdir2)
                    if os.path.isdir(subdir2_path):
                        image_folder = os.path.join(subdir2_path, 'imgs')
                        mask_folder = os.path.join(subdir2_path, 'mask')
                        self._read_img_mask(image_folder, mask_folder)

    def _read_img_mask(self, image_folder, mask_folder):
        for img_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, img_name.split('.')[0] + '.png')
            label_path = os.path.join(mask_folder, img_name.split('.')[0] + '.png')

            if os.path.exists(image_path) and os.path.exists(label_path):
                self.images.append(image_path)
                self.labels.append(label_path)
            else:
                print(f"Warning: Image or mask not found for {img_name}")

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)