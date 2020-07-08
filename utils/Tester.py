import os
from PIL import Image
from .log import logger

import numpy as np
import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from torchvision import transforms as T
from progress.bar import Bar

from .argmax import spatial_soft_argmax2d
from .evaluation import evalPCKh


class TestParams(object):
    # params based on your local env
    gpus = None  # default to use CPU mode

    # loading existing checkpoint
    ckpt = None     # path to the ckpt file
    testdata_dir = None
    savedata_dir = None
    subject_number = None

class Tester(object):

    TestParams = TestParams

    def __init__(self, model, test_params):
        assert isinstance(test_params, TestParams)
        self.params = test_params

        # load model
        self.model = model
        if len(self.params.gpus) > 0:
            gpus = ','.join([str(x) for x in self.params.gpus])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            self.params.gpus = tuple(range(len(self.params.gpus)))
            logger.info('Set CUDA_VISIBLE_DEVICES to GPU [{}]'.format(gpus))
            self.model = nn.DataParallel(
                self.model, device_ids=self.params.gpus)
            self.model = self.model.cuda()

        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        self.model.eval()


    def test(self):
        image_path, label_path = self._read_txt_file(self.params.testdata_dir)
        logger.info('Test dataset len: {}'.format(len(self.image_path)))

        bar = Bar('Testing', max=len(image_path))
        all_prediction = np.empty((0,2))
        all_target = np.empty((0,2))
        for item_number in range(len(image_path)):

            img = Image.open(image_path[item_number])
            img = tv_F.to_tensor(img)
            # T.Normalize(mean=[0.2017],std=[0.0817])
            # img = tv_F.to_tensor(tv_F.resize(img, (256, 256)))
            img = tv_F.normalize(img, mean=[0.2017],std=[0.0817])
            img_input = Variable(t.unsqueeze(img, 0))
            label = np.array(np.load(label_path[item_number])).reshape(-1, 2)
            with t.no_grad():

                if len(self.params.gpus) > 0:
                    img_input = img_input.cuda()

                output = self.model(img_input)

                predictions = spatial_soft_argmax2d(output[len(output)-1], 1000, False).cpu().numpy().reshape(-1, 2)
                predictions = 4*predictions + [1.5, 1.5] + [20, 120] 
                              
                all_prediction = np.concatenate((all_prediction, predictions), axis=0)
                all_target = np.concatenate((all_target, label), axis=0)
                
                steppckh, steperr = evalPCKh(predictions/4, label/4, threshold=50, alpha=0.2)
                bar.suffix = 'Test: [%(index)d/%(max)d] | PCKh: {pckh:6f} | AveErr: {err:.2f} pixel |'.format(pckh=steppckh, err=steperr)
            bar.next()
        bar.finish()

        all_prediction = all_prediction.reshape(self.params.subject_number, -1, 2)
        all_target = all_target.reshape(self.params.subject_number, -1, 2)
        for subject_item in range(self.params.subject_number):
            AvePCkh, AveErr = evalPCKh(all_prediction[subject_item]/4, all_target[subject_item]/4, threshold=50, alpha=0.2)
            print('Current subject : {}, test PCKh = {}, AveErr = {} pixel'.format(subject_item, AvePCkh, AveErr))

        if os.path.isdir(self.params.savedata_dir):
            pass
        else:
            os.makedirs(self.params.savedata_dir)
        np.save(self.params.savedata_dir + 'prediction.npy', all_prediction)


    def _read_txt_file(self, txt_path):
        self.image_path = []
        self.label_path = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(',')
                self.image_path.append(item[0])
                self.label_path.append(item[1])
            # print("Test dataset len: ", len(self.image_path))
        return self.image_path, self.label_path

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(t.load(ckpt))