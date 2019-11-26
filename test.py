from torch import nn
from utils import Tester
from network import *

# Set Test parameters
params = Tester.TestParams()
# set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.gpus = [0, 1]

params.ckpt = './models/256_2/ckpt_epoch_120.pth'
params.testdata_dir = './dataset/256/test_tiny.txt'

# models
model = StackedHourGlass(256, 1, 2, 4, 3)

# Test
tester = Tester(model, params)
tester.test()
