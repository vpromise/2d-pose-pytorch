import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils import Trainer
from dataset import CAREN
from network import StackedHourGlass


# Hyper-params
data_root = './dataset/right/'  # root for data path txt file
model_path = './models/right/'  # model save path
batch_size = 8
batch_size_valid = 1  # for valid
num_workers = 2

init_lr = 0.1
lr_decay = 0.8
momentum = 0.9
weight_decay = 0.000
nesterov = True

# Set Training parameters
params = Trainer.TrainParams()

params.max_epoch = 110
params.save_freq_epoch = 1
params.start_save_epoch = 1

params.criterion = nn.MSELoss()
params.gpus = [0, 1] #[0] [1] [0, 1]
params.save_dir = model_path
params.ckpt = None 
params.hm_type = 2 # [static, stage, liner, exp]


# load data
print("Loading dataset...")
train_data = CAREN(data_root, train=True)
val_data = CAREN(data_root, train=False)

batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)
batch_size_valid = batch_size_valid if len(params.gpus) == 0 else batch_size_valid*len(params.gpus)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))

val_dataloader = DataLoader(val_data, batch_size=batch_size_valid, shuffle=False, num_workers=1)
print('valid dataset len: {}'.format(len(val_dataloader.dataset)))

# models
model = StackedHourGlass(256, 2, 2, 3, 5)

# optimizer
trainable_vars = [param for param in model.parameters() if param.requires_grad]

params.optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)

# Train
params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=lr_decay, patience=1, cooldown=1, verbose=True)
trainer = Trainer(model, params, train_dataloader, val_dataloader)
trainer.train()