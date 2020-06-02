import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        if in_planes != out_planes:
            self.match = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.match = lambda x: x

    def forward(self, x):
        residual = self.match(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PoseResNet(nn.Module):
    def __init__(self, block, layers, keypoint_num = 5):
        self.inplanes = 64
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool2d = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.deconv_layer1 = nn.ConvTranspose2d(512 * block.expansion, 256, kernel_size = 4, stride = 2, padding = 1,
                                                bias = False)
        self.de_bn1 = nn.BatchNorm2d(256)
        self.deconv_layer2 = nn.ConvTranspose2d(256, 256, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.de_bn2 = nn.BatchNorm2d(256)
        self.deconv_layer3 = nn.ConvTranspose2d(256, 256, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.de_bn3 = nn.BatchNorm2d(256)

        self.final_layer = nn.Conv2d(256, keypoint_num, kernel_size = 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def _deconv_layer(self, num_layers, planes = 256):
        layers = []
        # planes
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(self.inplane, planes, kernel_size = 3, stride = 1, padding = 1,
                                   output_padding = 1, bias = False)
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace = True))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool2d(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
      
        out = F.relu(self.de_bn1(self.deconv_layer1(out)))
        out = F.relu(self.de_bn2(self.deconv_layer2(out)))
        out = F.relu(self.de_bn3(self.deconv_layer3(out)))
        
        out = self.final_layer(out)

        return out


def SimplePoseRes():
    '''
    resnet = {18: (BasicBlock, [2, 2, 2, 2]),
              34: (BasicBlock, [3, 4, 6, 3]),
              50: (Bottleneck, [3, 4, 6, 3]),
              101: (Bottleneck, [3, 4, 23, 3]),
              152: (Bottleneck, [3, 8, 36, 3])}
    '''
    # resnet50
    model = PoseResNet(Bottleneck, [3, 4, 6, 3])

    return model


if __name__ == '__main__':
    model = SimplePoseRes()
    print(model)
    data = torch.randn(3,1,480,320)
    out = model(data)
    # print(out)
    print(out.size())