# 2d-pose-caren.pytorch

Pytorch code for human pose estimation, especially for CAREN dataset.

## What's CAREN
The Computer Assisted Rehabilitation ENvironment (CAREN) is a versatile, multi sensory system for clinical analysis, rehabilitation, evaluation and registration of the human balance system. The use of virtual reality enables researchers to assess the subject’s behavior and includes sensory inputs like visual, auditory, vestibular and tactile.[see more.](https://www.motekforcelink.com/product/caren/)

## CAREN Data
- video
  - videos captured the movement of subject(usually patients) from three different angles, 50fps.
- csv
  - contains 3d location information of 21 markers attached to the subject's joints, per 0.01 second.
- c3d
  - contains original 3d location information of markers and some annotations about the information of platform and camera.
- report.pdf
  - the report document.

## Support Networks:

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

1. edit `pathgen.py` in dataset folder, change data_path to "/your/data/path/" and run `python dataset/pathgen.py`

2. run `tensorboard --logdir=runs` in terminal, open `tensorboard` for training visualization.

3. run `python train.py` start traing. 

## Test

1. `params.ckpt = './models/ckpt_epoch_100.pth'` 

2. `python test.py`
