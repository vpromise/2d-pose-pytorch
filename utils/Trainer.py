import math as m
import os
import time

import numpy as np
import torch as t
import torch.nn as nn
from progress.bar import Bar
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchnet import meter

from .log import logger


def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs


class TrainParams(object):
    # required params
    max_epoch = None

    # optimizer and criterion and learning rate scheduler
    optimizer = None
    criterion = None
    lr_scheduler = None  # should be an instance of ReduceLROnPlateau or _LRScheduler

    # params based on your local env
    gpus = []  # default to use CPU mode
    save_dir = None  # default `save_dir`
 
    # heatmap kernel size sigma
    hm_type = None

    # loading existing checkpoint
    ckpt = None  # path to the ckpt file

    # saving checkpoints
    save_freq_epoch = None  # save one ckpt per `save_freq_epoch` epochs
    start_save_epoch = None


class Trainer_refine(object):

    TrainParams = TrainParams

    def __init__(self, model, train_params, train_data, val_data=None):
        assert isinstance(train_params, TrainParams)
        self.params = train_params

        # Data loaders
        self.train_data = train_data
        self.val_data = val_data

        # criterion and Optimizer and learning rate and heatmap type
        self.last_epoch = 64
        self.hm_type = self.params.hm_type
        self.criterion = self.params.criterion
        self.optimizer = self.params.optimizer
        self.lr_scheduler = self.params.lr_scheduler
        
        logger.info('Set criterion to {}'.format(type(self.criterion)))
        logger.info('Set optimizer to {}'.format(type(self.optimizer)))
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))
        logger.info('Set heatmap kernel type to <{}>'.format(["static","stage","liner","exp"][int(self.hm_type)]))

        # load model
        self.model = model
        logger.info('Set output dir to {}'.format(self.params.save_dir))
        if os.path.isdir(self.params.save_dir):
            pass
        else:
            os.makedirs(self.params.save_dir)

        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # meters
        self.loss_meter = meter.AverageValueMeter()

        # tensorboard
        self.writer = SummaryWriter(max_queue=50)

        # set CUDA_VISIBLE_DEVICES
        if len(self.params.gpus) > 0:
            gpus = ','.join([str(x) for x in self.params.gpus])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            self.params.gpus = tuple(range(len(self.params.gpus)))
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpus))
            self.model = nn.DataParallel(
                self.model, device_ids=self.params.gpus)
            self.model = self.model.cuda()

        self.model.train()

    def train(self):

        best_loss = np.inf
        for epoch in range(self.last_epoch, self.params.max_epoch):

            self.loss_meter.reset()

            self.last_epoch += 1
            logger.info('Start training epoch {}'.format(self.last_epoch))

            # calculate trainng time for one epoch
            start_time = time.time()
            self._train_one_epoch()
            end_time = time.time()
            total_time = end_time - start_time

            # logger info: heatmap kernel sigma & training time
            logger.info('The heatmap sigma = {:.2f} pixel'.format(self.sigma))
            if total_time < 3600:
                logger.info('The training time = {:.2f} m {:.2f} s'.format(total_time//60, total_time%60))
            else:
                logger.info('The training time = {:.2f} h {:.2f} m {:.2f} s'.format(total_time//3600, (total_time % 3600)//60, (total_time % 3600)%60))

            # save model
            if (self.last_epoch >= self.params.start_save_epoch) and (self.last_epoch % self.params.save_freq_epoch == 0) or (self.last_epoch == self.params.max_epoch - 1):
                save_name = self.params.save_dir + 'ckpt_epoch_{}.pth'.format(self.last_epoch)
                t.save(self.model.state_dict(), save_name)

            # average_err
            average_err = self._val_one_epoch()
            logger.info('Joints ave_err = {:.4f} pixel'.format(average_err))

            # loss update
            if self.loss_meter.value()[0] < best_loss:
                logger.info('Found a better ckpt ({:.6f} -> {:.6f})'.format(best_loss, self.loss_meter.value()[0]))
                best_loss = self.loss_meter.value()[0]

            # tensorboard
            self.writer.add_scalar('info/hm_sigma', self.sigma, self.last_epoch)
            self.writer.add_scalar('info/train_loss', self.loss_meter.value()[0], self.last_epoch)
            self.writer.add_scalar('info/ave_err', average_err, self.last_epoch)

            # adjust the lr
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.loss_meter.value()[0], self.last_epoch)


    def _load_ckpt(self, ckpt):
        from collections import OrderedDict
        state_dict = t.load(ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    def _train_one_epoch(self):
        bar = Bar('Processing', max=len(self.train_data))
        for step, (data, label) in enumerate(self.train_data):
            bar.next()
            # train model
            # sigma for heatmap kernel size
            # self.hm_type = [static, stage, liner, exp] = [0, 1, 2, 3]
            if self.hm_type == 0:
                temp = 10
            elif self.hm_type == 1:
                # stage. parameter: k, b
                k, a, b = -30, 3, 9
                temp = int(self.last_epoch/k)*a + b
            elif self.hm_type == 2:
                # liner. parameter: k, b
                k, b = -0.1, 9
                temp = k*self.last_epoch + b
            elif self.hm_type == 3:
                # exp. parameter: [alpha beta k] or [a b k]
                a, b, k = -1/65, 3.2, 1
                temp = (m.exp(a*self.last_epoch)+k)**b
            self.sigma = max(temp,3)
            hm = []
            img_h, img_w = 120, 80
            for l_num in range(label.shape[0]):
                for j_num in range(label.shape[1]):
                    x, y = label[l_num][j_num][0], label[l_num][j_num][1]
                    hm.append(self._center_hm(img_h, img_w, np.array(x), np.array(y), self.sigma))
            hm = np.array(hm).reshape(data.size()[0], -1, img_h, img_w)
            inputs = Variable(data)
            target = Variable(t.from_numpy(hm))
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.type(t.FloatTensor).cuda()

            # forward
            score = self.model(inputs)
            loss = 0
            # stack hourglass
            # for s in range(len(score)):
            #     loss += self.criterion(score[s], target)
            # loss = loss/len(score)
            
            # simple pose res
            loss = self.criterion(score, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(None)

            # meters update
            self.loss_meter.add(loss.item())

        bar.finish()

    def _center_hm(self, img_h, img_w, c_x, c_y, sigma):
        X1 = np.linspace(1, img_h, img_h)
        Y1 = np.linspace(1, img_w, img_w)
        [Y, X] = np.meshgrid(Y1, X1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        self.heatmap = np.exp(-Exponent)
        return self.heatmap

    def _get_cord(self, heatmap):
        hm_size = heatmap.size()[1]
        [self.u, self.v] = [int(t.argmax(heatmap))//hm_size, int(t.argmax(heatmap)) % hm_size]
        return self.u, self.v

    def _val_one_epoch(self):
        self.model.eval()
        logger.info('Val on validation set...')
        all_err = []

        for step, (data, label) in enumerate(self.val_data):
            # val model
            with t.no_grad():
                inputs = data
                target = label.type(t.FloatTensor)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda(1)
                # target = target.cuda()
            score = self.model(inputs)
            err = []
            for i in range(target.size()[0]):
                for j in range(target.size()[1]):
                    # u, v = self._get_cord(score[0][i][j])
                    u, v = self._get_cord(score[i][j])
                    [x, y] = target[i][j].numpy()
                    err.append(np.sqrt(np.square(u-x) + np.square(v-y)))
            all_err.append(err)

            # print(inputs.size(), target.size(), score[1].shape)

            self.writer.add_image('img/in', inputs[0], self.last_epoch)

            # self.writer.add_image('img/tag_0', target[0][0:3], self.last_epoch)
            # self.writer.add_image('img/tag_1', target[0][2:5], self.last_epoch)
            # # self.writer.add_image('img/target_0_3', target[0][5:8], self.last_epoch)


            self.writer.add_image('img/out_0', score[0][0:3], self.last_epoch)
            self.writer.add_image('img/out_1', score[0][2:5], self.last_epoch)
            # self.writer.add_image('img_0/ouput_0_3', score[0][0][5:8], self.last_epoch)

        self.model.train()

        all_err = np.array(all_err)
        average_err = np.sum(all_err)/all_err.shape[1]
        return average_err