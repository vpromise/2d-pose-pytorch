# pose-caren.pytorch

human pose estimation in pytorch on CAREN datasets

## support networks:

- PoseAttention
- PyraNet
- StackedHourGlass

## Datasets

CAREN Dataset

## Requirements

- pytorch
- torchvision
- tensorboard

## Traing

`tensorboard --logdir=runs` 打开 `tensorboard` 面板,即可可视化训练过程

`python train.py` 即可

## Test

`params.ckpt = './models/ckpt_epoch_100.pth'` # 修改保存模型路径

`python test.py`