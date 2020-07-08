from torch import nn
from utils import Tester
from network import StackedHourGlass

# Set Test parameters
params = Tester.TestParams()
params.gpus = [0,1]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = '/media/vpromise/Inter/SavedModel/models/exp_new/ckpt_epoch_110.pth'
params.testdata_dir = './dataset/left/valid.txt'
params.savedata_dir = './saver/test/'
params.subject_number = 20
# models
model = StackedHourGlass(256, 2, 2, 3, 5)

# Test
tester = Tester(model, params)
tester.test()