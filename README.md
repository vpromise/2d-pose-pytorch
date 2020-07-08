# 2d-pose-caren.pytorch

Pytorch code for human pose estimation, especially for CAREN dataset.

## support networks:

- DeepPose
- PoseAttention
- PoseRes
- PyraNet
- StackedHourGlass

## Datasets

CAREN Dataset

## Requirements

- pytorch
- torchvision
- tensorboard

## Traing

1. edit `pathgen.py`, change data_path to "/your/data/path/" and run `python pathgen.py`

2. run `tensorboard --logdir=runs` in terminal, open `tensorboard` for training visualization.

3. run `python train.py` start traing. 

## Test

1. `params.ckpt = './models/ckpt_epoch_100.pth'` 

2. `python test.py`