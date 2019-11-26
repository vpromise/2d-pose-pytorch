# -*- coding: utf-8 -*-
"""
2D Human Pose Estimation for CAREN Data

Project by vpromise
Intelligent Healthcare Lab of UESTC
Created on 11 26 2019

@author: vpromise
@mail : vpromisever@gmail.com
@github : https://github.com/vpromise/
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class CAREN(data.Dataset):

    def __init__(self, root, transforms=None, train=True):
        '''
        Get images, divide into train/val set
        '''
        self.train = train
        self.data_root = root
        self._read_txt_file()

        if transforms is None:
            # normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
            #                         std=[0.229, 0.224, 0.225])

            if not train:
                self.transforms = T.Compose([
                    T.Resize((256, 256)),
                    T.ToTensor(),
                    # normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize((256, 256)),
                    T.ToTensor(),
                    # normalize
                ])

    def _read_txt_file(self):
        '''
        Get images and heatmap path
        '''
        self.image_path = []
        self.heatmap_path = []

        if self.train:
            txt_file = self.data_root + "train.txt"
        else:
            txt_file = self.data_root + "valid.txt"

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(',')
                self.image_path.append(item[0])
                self.heatmap_path.append(item[1])

    def __getitem__(self, index):
        '''
        Return the data and heatmap label of one image 
        '''
        img_path = self.image_path[index]
        hm_path = self.heatmap_path[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        label = np.array(np.load(hm_path))
        
        return data, label

    def __len__(self):
        return len(self.image_path)
