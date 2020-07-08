import numpy
import torch
from PIL import Image
from torchvision import transforms as T

txt_file = "./dataset/right/train.txt"
image_path = []
with open(txt_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        item = line.strip().split(',')
        image_path.append(item[0])

data_len = len(image_path)
print('All dataset len:', data_len)

transforms = T.Compose([T.ToTensor()])

mean = 0.0
std = 0.0
for inx in range(len(image_path)):
    data = Image.open(image_path[inx])
    data = transforms(data)
    mean += data.mean()
    std += data.std()
    if inx % 1000 == 0:
        print('success calculate {:6} images'.format(inx))

mean = mean/data_len
std = std/data_len
# numpy.save('./dataset/right/img_mean_std.npy', [mean, std])

with open('./dataset/right/img_mean_std.txt', 'w') as f:
    f.write(str(mean) + '\n' + str(std))

print('mean = ', mean)
print('std = ', std)
