from __future__ import print_function

import os

import numpy as np
import torch as t
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms as T

from .log import logger


class TestParams(object):
    # params based on your local env
    gpus = []  # default to use CPU mode

    # loading existing checkpoint
    ckpt = './models/256/ckpt_epoch_120.pth'     # path to the ckpt file

    testdata_dir = './dataset/256/test.txt'

class Tester(object):

    TestParams = TestParams

    def __init__(self, model, test_params):
        assert isinstance(test_params, TestParams)
        self.params = test_params

        # load model
        self.model = model
        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES, 1 GPU is enough
        if len(self.params.gpus) > 0:
            gpu_test = str(self.params.gpus[0])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpu_test))
            self.model = self.model.cuda()

        self.model.eval()



    def test(self):

        image_path, heatmap_path = self._read_txt_file(self.params.testdata_dir)

        all_prediction = []
        for img_number in range(len(image_path)):
            print('Processing image: ' + image_path[img_number])

            img = Image.open(image_path[img_number])
            img = tv_F.to_tensor(tv_F.resize(img, (256, 256)))
            # img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(t.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()

            output = self.model(img_input)[0][0]

            prediction = []
            for k in range(3):
                [prediction_x, prediction_y] = self._get_cord(output[k])
                prediction_x = 1.25*prediction_x + 20
                prediction_y = 1.25*prediction_y + 280

                prediction.append([prediction_x, prediction_y])
            # print("Prediction joints location is : ", prediction)
            all_prediction.append(prediction)
        # save predicted 2d joints location
        np.save('./prediction_256.npy', all_prediction)
    
    def _get_cord(self, heatmap):
        # assert heatmap.size() == (64,64)
        heatmap_size = heatmap.size()[1]
        [self.u, self.v] = [int(t.argmax(heatmap)) % heatmap_size,  int(t.argmax(heatmap)) // heatmap_size]
        return self.u, self.v

    def _read_txt_file(self, txt_path):
        self.image_path = []
        self.heatmap_path = []

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(',')
                self.image_path.append(item[0])
                self.heatmap_path.append(item[1])
            print("test dataset len: ", len(self.image_path))
        return self.image_path, self.heatmap_path

    def _load_ckpt(self, ckpt):
        from collections import OrderedDict
        state_dict = t.load(ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
