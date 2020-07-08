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
# from .toolkit import load_ckpt
from .heatmap import hm_kernel_size, gene_heatmap
from .argmax import hm_argmax, soft_argmax, spatial_soft_argmax2d
from .evaluation import evalPCKh


def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs


class TrainParams(object):
    # required params
    # edit in train.py file is ok
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
    ckpt = None  

    # saving checkpoints
    save_freq_epoch = None  # save one ckpt per `save_freq_epoch` epochs
    start_save_epoch = None


class Trainer(object):

    TrainParams = TrainParams

    def __init__(self, model, train_params, train_data, val_data=None):
        assert isinstance(train_params, TrainParams)
        self.params = train_params

        # Data loaders
        self.train_data = train_data
        self.val_data = val_data

        # Criterion, Optimizer, learning rate and heatmap type init
        self.last_epoch = 0
        self.hm_type = self.params.hm_type
        self.criterion = self.params.criterion
        self.optimizer = self.params.optimizer
        self.lr_scheduler = self.params.lr_scheduler

        logger.info('Set criterion to {}'.format(type(self.criterion)))
        logger.info('Set optimizer to {}'.format(type(self.optimizer)))
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))
        logger.info('Set heatmap refine to <{}>'.format(["static", "stage", "liner", "exp", "new"][int(self.hm_type)]))

        # load model
        self.model = model

        # set CUDA_VISIBLE_DEVICES
        if len(self.params.gpus) > 0:
            gpus = ','.join([str(x) for x in self.params.gpus])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            self.params.gpus = tuple(range(len(self.params.gpus)))
            logger.info('Set CUDA_VISIBLE_DEVICES to GPU[{}]'.format(gpus))
            self.model = nn.DataParallel(self.model, device_ids=self.params.gpus)
            self.model = self.model.cuda()

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

        # train
        self.model.train()


    def train(self):

        best_loss = np.inf
        for epoch in range(self.last_epoch, self.params.max_epoch):

            self.loss_meter.reset()

            epoch += 1
            self.last_epoch += 1
            print(' ')
            logger.info('Start training epoch {}'.format(epoch))

            # calculate trainng time for one epoch
            start_time = time.time()
            self._train_one_epoch()
            total_time = time.time() - start_time

            # logger info: heatmap kernel sigma & training time
            logger.info('The heatmap kernel size = {:.2f} pixel'.format(self.sigma))
            logger.info('The training time = {:.2f} m {:.2f} s'.format(total_time//60, total_time % 60))

            # save model
            if (epoch >= self.params.start_save_epoch) and (epoch % self.params.save_freq_epoch == 0) or (epoch == self.params.max_epoch - 1):
                save_name = self.params.save_dir + 'ckpt_epoch_{}.pth'.format(epoch)
                t.save(self.model.state_dict(), save_name)

            # validate and get average_err
            logger.info('Val on validation set...')
            self._val_one_epoch()
            logger.info('Mean Per Joint 2D Error = {:.4f} pixel'.format(self.AveErr))

            # loss update
            if self.loss_meter.value()[0] < best_loss:
                logger.info('Found a better ckpt ({:.6f} -> {:.6f})'.format(best_loss, self.loss_meter.value()[0]))
                best_loss = self.loss_meter.value()[0]

            # tensorboard
            self.writer.add_scalar('train/hm_kernel', self.sigma, self.last_epoch)
            self.writer.add_scalar('train/loss', self.loss_meter.value()[0], self.last_epoch)
            self.writer.add_scalar('train/ave_err', self.AveErr, self.last_epoch)
            self.writer.add_scalar('train/PCHk', self.PCkh, self.last_epoch)

            # adjust the lr
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.loss_meter.value()[0])


    def _train_one_epoch(self):
        bar = Bar('Processing', max=len(self.train_data))
        for step, (data, label) in enumerate(self.train_data):

            self.sigma = hm_kernel_size(self.hm_type, self.last_epoch, threshold=4)
            target = gene_heatmap(label, self.sigma)
            inputs = Variable(data)
            target = Variable(t.from_numpy(target))
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.type(t.FloatTensor).cuda()

            # forward
            score = self.model(inputs) 
            loss = 0

            # stack hourglass
            for s in range(len(score)):
                loss += self.criterion(score[s], target)
            loss = loss/len(score)

            # simple pose res
            # loss = self.criterion(score[1], target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(None)

            # meters update
            self.loss_meter.add(loss.item())
            
            # evaluation: calculate PCKh
            predictions = spatial_soft_argmax2d(score[len(score)-1], 1000, False).cpu().numpy().reshape(-1, 2)
            targetcoors = label.numpy().reshape(-1,2)
            steppckh, steperr = evalPCKh(predictions, targetcoors, threshold=50, alpha=0.2)

            # tensorboard show
            if step % 500 == 0:
                target_shows = t.sum(target[0], 0)
                target_shows[target_shows > 1] = 1
                self.writer.add_image('train/input', inputs[0], self.last_epoch)
                self.writer.add_image('train/taget', target_shows, self.last_epoch, dataformats='HW')
                self.writer.add_image('train/output', t.sum(score[1][0], 0), self.last_epoch, dataformats='HW')

            bar.suffix = 'Train: [%(index)d/%(max)d] | Epoch: [{0}/{1}]| Loss: {loss:6f} | PCKh: {pckh:4f} | AveErr: {err:.2f} pixel |'.format(self.last_epoch, self.params.max_epoch, loss=loss, pckh=steppckh, err=steperr)
            bar.next()
        bar.finish()

    def _val_one_epoch(self):
        bar = Bar('Validating', max=len(self.val_data))
        self.model.eval()

        predictions = np.empty((0, 2))
        targetcoors = np.empty((0, 2))

        for step, (data, label) in enumerate(self.val_data):
            with t.no_grad():
                inputs = data
                target = label.reshape(-1, 2)
                # target = label.type(t.FloatTensor)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda(1)
                # target = target.cuda()

            score = self.model(inputs)
            coors = spatial_soft_argmax2d(score[len(score)-1], 1000, False).cpu().numpy().reshape(-1, 2)
            predictions = np.concatenate((predictions, coors), axis=0)
            targetcoors = np.concatenate((targetcoors, target), axis=0)

            # evaluation: calculate PCKh
            currentpckh, currenterr = evalPCKh(predictions, targetcoors, threshold=50, alpha=0.2)

            # tensorboard visualization
            if step % 100 == 0:
                self.writer.add_image('valid/img', inputs[0], self.last_epoch)
                self.writer.add_image('valid/output', t.sum(score[1][0], 0), self.last_epoch, dataformats='HW')

            bar.suffix = 'Valid: [%(index)d/%(max)d] | PCKh: {pckh:6f} | AveErr: {err:.2f} pixel |'.format(pckh=currentpckh, err=currenterr)
            bar.next()
        bar.finish()
        
        self.PCkh, self.AveErr = evalPCKh(predictions, targetcoors, threshold=50, alpha=0.2)
        self.model.train()
        

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(t.load(ckpt))