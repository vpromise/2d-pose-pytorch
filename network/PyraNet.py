import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import models.modules.PyraNet as M

class BnReluConv(nn.Module):
	"""docstring for BnReluConv"""
	def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
		super(BnReluConv, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.kernelSize = kernelSize
		self.stride = stride
		self.padding = padding

		self.bn = nn.BatchNorm2d(self.inChannels)
		self.relu = nn.ReLU()
		self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)

	def forward(self, x):
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv(x)
		return x

class Pyramid(nn.Module):
	"""docstring for Pyramid"""
	def __init__(self, D, cardinality, inputRes):
		super(Pyramid, self).__init__()
		self.D = D
		self.cardinality = cardinality
		self.inputRes = inputRes
		self.scale = 2**(-1/self.cardinality)
		_scales = []
		for card in range(self.cardinality):
			temp = nn.Sequential(
					nn.FractionalMaxPool2d(2, output_ratio = self.scale**(card + 1)),
					nn.Conv2d(self.D, self.D, 3, 1, 1),
					nn.Upsample(size = self.inputRes)#, mode='bilinear')
				)
			_scales.append(temp)
		self.scales = nn.ModuleList(_scales)

	def forward(self, x):
		#print(x.shape, self.inputRes)
		out = torch.zeros_like(x)
		for card in range(self.cardinality):
			out += self.scales[card](x)
		return out

class BnReluPyra(nn.Module):
	"""docstring for BnReluPyra"""
	def __init__(self, D, cardinality, inputRes):
		super(BnReluPyra, self).__init__()
		self.D = D
		self.cardinality = cardinality
		self.inputRes = inputRes
		self.bn = nn.BatchNorm2d(self.D)
		self.relu = nn.ReLU()
		self.pyra = Pyramid(self.D, self.cardinality, self.inputRes)

	def forward(self, x):
		x = self.bn(x)
		x = self.relu(x)
		x = self.pyra(x)
		return x


class ConvBlock(nn.Module):
	"""docstring for ConvBlock"""
	def __init__(self, inChannels, outChannels):
		super(ConvBlock, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.outChannelsby2 = outChannels//2

		self.cbr1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
		self.cbr2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
		self.cbr3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

	def forward(self, x):
		x = self.cbr1(x)
		x = self.cbr2(x)
		x = self.cbr3(x)
		return x

class PyraConvBlock(nn.Module):
	"""docstring for PyraConvBlock"""
	def __init__(self, inChannels, outChannels, inputRes, baseWidth, cardinality, type = 1):
		super(PyraConvBlock, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.inputRes = inputRes
		self.baseWidth = baseWidth
		self.cardinality = cardinality
		self.outChannelsby2 = outChannels//2
		self.D = self.outChannels // self.baseWidth
		self.branch1 = nn.Sequential(
				BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0),
				BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
			)
		self.branch2 = nn.Sequential(
				BnReluConv(self.inChannels, self.D, 1, 1, 0),
				BnReluPyra(self.D, self.cardinality, self.inputRes),
				BnReluConv(self.D, self.outChannelsby2, 1, 1, 0)
			)
		self.afteradd = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

	def forward(self, x):
		x = self.branch2(x) + self.branch1(x)
		x = self.afteradd(x)
		return x

class SkipLayer(nn.Module):
	"""docstring for SkipLayer"""
	def __init__(self, inChannels, outChannels):
		super(SkipLayer, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		if (self.inChannels == self.outChannels):
			self.conv = None
		else:
			self.conv = nn.Conv2d(self.inChannels, self.outChannels, 1)

	def forward(self, x):
		if self.conv is not None:
			x = self.conv(x)
		return x

class Residual(nn.Module):
	"""docstring for Residual"""
	def __init__(self, inChannels, outChannels, inputRes=None, baseWidth=None, cardinality=None, type=None):
		super(Residual, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cb = ConvBlock(self.inChannels, self.outChannels)
		self.skip = SkipLayer(self.inChannels, self.outChannels)

	def forward(self, x):
		out = 0
		out = out + self.cb(x)
		out = out + self.skip(x)
		return out

class ResidualPyramid(nn.Module):
	"""docstring for ResidualPyramid"""
	def __init__(self, inChannels, outChannels, inputRes, baseWidth, cardinality, type = 1):
		super(ResidualPyramid, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.inputRes = inputRes
		self.baseWidth = baseWidth
		self.cardinality = cardinality
		self.type = type
		self.cb = PyraConvBlock(self.inChannels, self.outChannels, self.inputRes, self.baseWidth, self.cardinality, self.type)
		self.skip = SkipLayer(self.inChannels, self.outChannels)

	def forward(self, x):
		out = 0
		out = out + self.cb(x)
		out = out + self.skip(x)
		return out


### PyraNet

class PyraNetHourGlass(nn.Module):
	"""docstring for PyraNetHourGlass"""
	def __init__(self, nChannels=256, numReductions=4, nModules=2, inputRes=256, baseWidth=6, cardinality=30, poolKernel=(2,2), poolStride=(2,2), upSampleKernel=2):
		super(PyraNetHourGlass, self).__init__()
		self.numReductions = numReductions
		self.nModules = nModules
		self.nChannels = nChannels
		self.poolKernel = poolKernel
		self.poolStride = poolStride
		self.upSampleKernel = upSampleKernel

		self.inputRes = inputRes
		self.baseWidth = baseWidth
		self.cardinality = cardinality
		"""
		For the skip connection, a residual module (or sequence of residuaql modules)
		"""
		Residualskip = ResidualPyramid if numReductions > 1 else Residual
		Residualmain = ResidualPyramid if numReductions > 2 else Residual
		_skip = []
		for _ in range(self.nModules):
			_skip.append(Residualskip(self.nChannels, self.nChannels, self.inputRes, self.baseWidth, self.cardinality))

		self.skip = nn.Sequential(*_skip)

		"""
		First pooling to go to smaller dimension then pass input through
		Residual Module or sequence of Modules then  and subsequent cases:
			either pass through Hourglass of numReductions-1
			or pass through Residual Module or sequence of Modules
		"""

		self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

		_afterpool = []
		for _ in range(self.nModules):
			_afterpool.append(Residualmain(self.nChannels, self.nChannels, self.inputRes//2, self.baseWidth, self.cardinality))

		self.afterpool = nn.Sequential(*_afterpool)

		if (numReductions > 1):
			self.hg = PyraNetHourGlass(self.nChannels, self.numReductions-1, self.nModules, self.inputRes//2, self.baseWidth, self.cardinality, self.poolKernel, self.poolStride, self.upSampleKernel)
		else:
			_num1res = []
			for _ in range(self.nModules):
				_num1res.append(Residualmain(self.nChannels,self.nChannels, self.inputRes//2, self.baseWidth, self.cardinality))

			self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

		"""
		Now another Residual Module or sequence of Residual Modules
		"""

		_lowres = []
		for _ in range(self.nModules):
			_lowres.append(Residualmain(self.nChannels,self.nChannels, self.inputRes//2, self.baseWidth, self.cardinality))

		self.lowres = nn.Sequential(*_lowres)

		"""
		Upsampling Layer (Can we change this??????)
		As per Newell's paper upsamping recommended
		"""
		self.up = nn.Upsample(scale_factor = self.upSampleKernel)


	def forward(self, x):
		out1 = x
		out1 = self.skip(out1)
		out2 = x
		out2 = self.mp(out2)
		out2 = self.afterpool(out2)
		if self.numReductions>1:
			out2 = self.hg(out2)
		else:
			out2 = self.num1res(out2)
		out2 = self.lowres(out2)
		out2 = self.up(out2)

		return out2 + out1


class PyraNet(nn.Module):
	"""docstring for PyraNet"""
	def __init__(self, nChannels=256, nStack=4, nModules=2, numReductions=4, baseWidth=6, cardinality=30, nJoints=16, inputRes=256):
		super(PyraNet, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		self.baseWidth = baseWidth
		self.cardinality = cardinality
		self.inputRes = inputRes
		self.nJoints = nJoints

		self.start = BnReluConv(1, 64, kernelSize = 7, stride = 1, padding = 3)


		self.res1 = ResidualPyramid(64, 128, self.inputRes, self.baseWidth, self.cardinality, 0)
		self.mp = nn.MaxPool2d(2, 2)
		self.res2 = ResidualPyramid(128, 128, self.inputRes//2, self.baseWidth, self.cardinality,)
		self.res3 = ResidualPyramid(128, self.nChannels, self.inputRes//2, self.baseWidth, self.cardinality)

		_hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]

		for _ in range(self.nStack):
			_hourglass.append(PyraNetHourGlass(self.nChannels, self.numReductions, self.nModules, self.inputRes//2, self.baseWidth, self.cardinality))
			_ResidualModules = []
			for _ in range(self.nModules):
				_ResidualModules.append(Residual(self.nChannels, self.nChannels))
			_ResidualModules = nn.Sequential(*_ResidualModules)
			_Residual.append(_ResidualModules)
			_lin1.append(BnReluConv(self.nChannels, self.nChannels))
			_chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints,1))
			_lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
			_jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

		self.hourglass = nn.ModuleList(_hourglass)
		self.Residual = nn.ModuleList(_Residual)
		self.lin1 = nn.ModuleList(_lin1)
		self.chantojoints = nn.ModuleList(_chantojoints)
		self.lin2 = nn.ModuleList(_lin2)
		self.jointstochan = nn.ModuleList(_jointstochan)

	def forward(self, x):
		x = self.start(x)
		x = self.res1(x)
		x = self.mp(x)
		x = self.res2(x)
		x = self.res3(x)
		out = []

		for i in range(self.nStack):
			x1 = self.hourglass[i](x)
			x1 = self.Residual[i](x1)
			x1 = self.lin1[i](x1)
			out.append(self.chantojoints[i](x1))
			x1 = self.lin2[i](x1)
			x = x + x1 + self.jointstochan[i](out[i])

		return (out)

# import numpy as np

# if __name__ == "__main__":
#     Net = PyraNet(nChannels=256, nStack=1, nModules=2, numReductions=4, baseWidth=6, cardinality=30, nJoints=3, inputRes=256)
#     Net = Net.double()


#     x  = np.random.randn(1, 1, 256,256)
#     xt = torch.from_numpy(x)
#     xt = xt.clone().detach()

#     y = Net(xt)
#     y = np.array(y)
#     print(y.shape)
#     print(y[0].shape)
