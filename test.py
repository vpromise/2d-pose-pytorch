from torch import nn
from utils import Tester
from network import *

# Set Test parameters
params = Tester.TestParams()
params.gpus = [1]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
# params.ckpt = './models/size/8.5/ckpt_epoch_1.pth' 
params.ckpt = './models/refine_exp/ckpt_epoch_100.pth'
params.testdata_dir = './dataset/refine/test_inf.txt'

# models
model = StackedHourGlass(256, 2, 2, 3, 5)

# Test
tester = Tester(model, params)
tester.test()