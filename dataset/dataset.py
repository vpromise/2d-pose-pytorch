# -*- coding: utf-8 -*-
"""
2D Human Pose Estimation for CAREN Data

Project by vpromise
Intelligent Healthcare Lab of UESTC
Created on 11/26/2019
Update on 02/06/2020

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

            # for caren gray img, mean = 0.2017, std = 0.0817
            normalize = T.Normalize(mean=[0.2017],
                                    std=[0.0817])

            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])

    def _read_txt_file(self):
        self.image_path = []
        self.label_path = []

        if self.train:
            txt_file = self.data_root + "train.txt"
        else:
            txt_file = self.data_root + "valid.txt"

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(',')
                self.image_path.append(item[0])
                self.label_path.append(item[1])

    def __getitem__(self, index):
        '''
        return the data of one image [1,h,w]
        return the label of joints for generating heatmap
        label [jointss,2] --> hm [joint, h/4, w/4] 
        '''
        img_path = self.image_path[index]
        lab_path = self.label_path[index]
        data = Image.open(img_path)
        data = self.transforms(data)

        lab = np.array(np.load(lab_path)).reshape(-1,2)
        y, x = lab[:,0].reshape(-1,1),lab[:,1].reshape(-1,1)
        x = (x - 120)/4

        # edit joint location to satisfied img crop 
        if lab_path[-10:-9] == '0':
            y = (y - 20)/4
        elif lab_path[-10:-9] == '2':
            y = (y - 110)/4
        elif lab_path[-10:-9] == '1':
            # x = (x - 120)/4
            y = (y - 110)/4
        label = np.hstack((x, y))
        return data, label

    def __len__(self):
        return len(self.image_path)