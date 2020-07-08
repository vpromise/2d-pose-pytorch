# -*- coding:utf-8 -*-
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
                    normalize])
            else:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize])

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
        label = np.array(np.load(lab_path)).reshape(-1, 2)

        data_type = ['left', 'right', 'all']
        if data_type[0] in lab_path:
            label = label-[20, 120]
        elif data_type[1] in lab_path:
            label = label-[110, 120]
        elif data_type[2] in lab_path:
            label = label-[110, 120]

        label = label/4
        return data, label

    def __len__(self):
        return len(self.image_path)