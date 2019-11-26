import os

import numpy as np
import torch as t
import torch.nn as nn
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
    save_dir = './models/'  # default `save_dir`

    # loading existing checkpoint
    ckpt = None  # path to the ckpt file

    # saving checkpoints
    save_freq_epoch = None  # save one ckpt per `save_freq_epoch` epochs
    start_save_epoch = 50


class Trainer(object):

    TrainParams = TrainParams

    def __init__(self, model, train_params, train_data, val_data=None):
        assert isinstance(train_params, TrainParams)
        self.params = train_params

        # Data loaders
        self.train_data = train_data
        self.val_data = val_data

        # criterion and Optimizer and learning rate
        self.last_epoch = 0
        self.criterion = self.params.criterion
        self.optimizer = self.params.optimizer
        self.lr_scheduler = self.params.lr_scheduler
        logger.info('Set criterion to {}'.format(type(self.criterion)))
        logger.info('Set optimizer to {}'.format(type(self.optimizer)))
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))

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
        self.writer = SummaryWriter()

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

            self._train_one_epoch()

            # save model
            if (self.last_epoch >= self.params.start_save_epoch) and (self.last_epoch % self.params.save_freq_epoch == 0) or (self.last_epoch == self.params.max_epoch - 1):

                save_name = self.params.save_dir + \
                    'ckpt_epoch_{}.pth'.format(self.last_epoch)
                t.save(self.model.state_dict(), save_name)

            average_err = self._val_one_epoch()
            logger.info(
                'Joint average err is {:.4f} pixel'.format(average_err))

            if self.loss_meter.value()[0] < best_loss:
                logger.info(
                    'Found a better ckpt ({:.6f} -> {:.6f})'.format(best_loss, self.loss_meter.value()[0]))
                best_loss = self.loss_meter.value()[0]

            # tensorboard
            self.writer.add_scalar(
                'info/train_loss', self.loss_meter.value()[0], self.last_epoch)
            self.writer.add_scalar(
                'info/average_err', average_err, self.last_epoch)

            # adjust the lr
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.loss_meter.value()[
                                       0], self.last_epoch)

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(t.load(ckpt))

    def _train_one_epoch(self):
        for step, (data, label) in enumerate(self.train_data):
            # train model
            inputs = Variable(data)
            target = Variable(label)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.type(t.FloatTensor).cuda()

            # forward
            score = self.model(inputs)
            loss = 0
            for s in range(len(score)):
                loss += self.criterion(score[s], target)
            # loss = self.criterion(score[0], target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(None)
            # meters update
            self.loss_meter.add(loss.item())

    def _get_cord(self, heatmap):
        heatmap_size = heatmap.size()[1]
        [self.u, self.v] = [
            int(t.argmax(heatmap))//heatmap_size, int(t.argmax(heatmap)) % heatmap_size]
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
                inputs = inputs.cuda()
                target = target.cuda()

            score = self.model(inputs)

            err = []
            for i in range(target.size()[0]):
                for j in range(target.size()[1]):
                    u, v = self._get_cord(score[0][i][j])
                    x, y = self._get_cord(target[i][j])
                    err.append(np.sqrt(np.square(u-x) + np.square(v-y)))
            all_err.append(err)

            print(inputs.size(), target.size(), score[0].shape)
            self.writer.add_image('val_0/input', inputs[0], self.last_epoch)
            self.writer.add_image('val_0/heatmap', target[0], self.last_epoch)
            self.writer.add_image('val_0/ouput', score[0][0], self.last_epoch)

            self.writer.add_image('val_1/input', inputs[1], self.last_epoch)
            self.writer.add_image('val_1/heatmap', target[1], self.last_epoch)
            self.writer.add_image('val_1/ouput', score[0][1], self.last_epoch)

        self.model.train()

        all_err = np.array(all_err)
        # print(all_err, all_err.shape)
        average_err = np.sum(all_err)/all_err.shape[1]
        return average_err
