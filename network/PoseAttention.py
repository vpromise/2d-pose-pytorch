import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# import models.modules.PoseAttention as M




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
		self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv(x)
		return x

class BnReluPoolConv(nn.Module):
		"""docstring for BnReluPoolConv"""
		def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
			super(BnReluPoolConv, self).__init__()
			self.inChannels = inChannels
			self.outChannels = outChannels
			self.kernelSize = kernelSize
			self.stride = stride
			self.padding = padding

			self.bn = nn.BatchNorm2d(self.inChannels)
			self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
			self.relu = nn.ReLU()

		def forward(self, x):
			x = self.bn(x)
			x = self.relu(x)
			x = F.max_pool2d(x, kernel_size=2, stride=2)
			x = self.conv(x)
			return x

class ConvBlock(nn.Module):
	"""docstring for ConvBlock"""
	def __init__(self, inChannels, outChannels):
		super(ConvBlock, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.outChannelsby2 = outChannels//2

		self.brc1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
		self.brc2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
		self.brc3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

	def forward(self, x):
		x = self.brc1(x)
		x = self.brc2(x)
		x = self.brc3(x)
		return x

class PoolConvBlock(nn.Module):
	"""docstring for PoolConvBlock"""
	def __init__(self, inChannels, outChannels):
		super(PoolConvBlock, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels

		self.brpc = BnReluPoolConv(self.inChannels, self.outChannels, 3, 1, 1)
		self.brc = BnReluConv(self.outChannels, self.outChannels, 3, 1, 1)

	def forward(self, x):
		x = self.brpc(x)
		x = self.brc(x)
		x = F.interpolate(x, scale_factor=2)
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
	def __init__(self, inChannels, outChannels):
		super(Residual, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cb = ConvBlock(inChannels, outChannels)
		self.skip = SkipLayer(inChannels, outChannels)

	def forward(self, x):
		out = 0
		out = out + self.cb(x)
		out = out + self.skip(x)
		return out

class HourGlassResidual(nn.Module):
	"""docstring for HourGlassResidual"""
	def __init__(self, inChannels, outChannels):
		super(HourGlassResidual, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cb = ConvBlock(inChannels, outChannels)
		self.pcb = PoolConvBlock(inChannels, outChannels)
		self.skip = SkipLayer(inChannels, outChannels)


	def forward(self, x):
		out = 0
		out = out + self.cb(x)
		out = out + self.pcb(x)
		out = out + self.skip(x)
		return out

class AttentionIter(nn.Module):
	"""docstring for AttentionIter"""
	def __init__(self, nChannels, LRNSize, IterSize):
		super(AttentionIter, self).__init__()
		self.nChannels = nChannels
		self.LRNSize = LRNSize
		self.IterSize = IterSize
		self.bn = nn.BatchNorm2d(self.nChannels)
		self.U = nn.Conv2d(self.nChannels, 1, 3, 1, 1)
		# self.spConv = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		# self.spConvclone = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		# self.spConvclone.load_state_dict(self.spConv.state_dict())
		_spConv_ = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		_spConv = []
		for i in range(self.IterSize):
			_temp_ = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
			_temp_.load_state_dict(_spConv_.state_dict())
			_spConv.append(nn.BatchNorm2d(1))
			_spConv.append(_temp_)
		self.spConv = nn.ModuleList(_spConv)

	def forward(self, x):
		x = self.bn(x)
		u = self.U(x)
		out = u
		for i in range(self.IterSize):
			# if (i==1):
			# 	out = self.spConv(out)
			# else:
			# 	out = self.spConvclone(out)
			out = self.spConv[2*i](out)
			out = self.spConv[2*i+1](out)
			out = u + torch.sigmoid(out)
		return (x * out.expand_as(x))

class AttentionPartsCRF(nn.Module):
	"""docstring for AttentionPartsCRF"""
	def __init__(self, nChannels, LRNSize, IterSize, nJoints):
		super(AttentionPartsCRF, self).__init__()
		self.nChannels = nChannels
		self.LRNSize = LRNSize
		self.IterSize = IterSize
		self.nJoints = nJoints
		_S = []
		for _ in range(self.nJoints):
			_S_ = []
			_S_.append(AttentionIter(self.nChannels, self.LRNSize, self.IterSize))
			_S_.append(nn.BatchNorm2d(self.nChannels))
			_S_.append(nn.Conv2d(self.nChannels, 1, 1, 1, 0))
			_S.append(nn.Sequential(*_S_))
		self.S = nn.ModuleList(_S)

	def forward(self, x):
		out = []
		for i in range(self.nJoints):
			#out.append(self.S[i](self.attiter(x)))
			out.append(self.S[i](x))
		return torch.cat(out, 1)





class HourglassAttention(nn.Module):
	"""docstring for HourglassAttention"""
	def __init__(self, nChannels = 256, numReductions = 4, nModules = 2, poolKernel = (2,2), poolStride = (2,2), upSampleKernel = 2):
		super(HourglassAttention, self).__init__()
		self.numReductions = numReductions
		self.nModules = nModules
		self.nChannels = nChannels
		self.poolKernel = poolKernel
		self.poolStride = poolStride
		self.upSampleKernel = upSampleKernel
		"""
		For the skip connection, a Residual module (or sequence of residuaql modules)
		"""

		_skip = []
		for _ in range(self.nModules):
			_skip.append(Residual(self.nChannels, self.nChannels))

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
			_afterpool.append(Residual(self.nChannels, self.nChannels))

		self.afterpool = nn.Sequential(*_afterpool)

		if (numReductions > 1):
			self.hg = HourglassAttention(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
		else:
			_num1res = []
			for _ in range(self.nModules):
				_num1res.append(Residual(self.nChannels,self.nChannels))

			self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

		"""
		Now another Residual Module or sequence of Residual Modules
		"""

		_lowres = []
		for _ in range(self.nModules):
			_lowres.append(Residual(self.nChannels,self.nChannels))

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


class PoseAttention(nn.Module):
	"""docstring for PoseAttention"""
	def __init__(self, nChannels, nStack, nModules, numReductions, nJoints, LRNSize, IterSize):
		super(PoseAttention, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		self.nJoints = nJoints
		self.LRNSize = LRNSize
		self.IterSize = IterSize
		self.nJoints = nJoints

		self.start = BnReluConv(3, 64, kernelSize = 7, stride = 2, padding = 3)

		self.res1 = Residual(64, 128)
		self.mp = nn.MaxPool2d(2, 2)
		self.res2 = Residual(128, 128)
		self.res3 = Residual(128, self.nChannels)

		_hourglass, _Residual, _lin1, _attiter, _chantojoints, _lin2, _jointstochan = [], [],[],[],[],[],[]

		for i in range(self.nStack):
			_hourglass.append(HourglassAttention(self.nChannels, self.numReductions, self.nModules))
			_ResidualModules = []
			for _ in range(self.nModules):
				_ResidualModules.append(Residual(self.nChannels, self.nChannels))
			_ResidualModules = nn.Sequential(*_ResidualModules)
			_Residual.append(_ResidualModules)
			_lin1.append(BnReluConv(self.nChannels, self.nChannels))
			_attiter.append(AttentionIter(self.nChannels, self.LRNSize, self.IterSize))
			if i<self.nStack//2:
				_chantojoints.append(
						nn.Sequential(
							nn.BatchNorm2d(self.nChannels),
							nn.Conv2d(self.nChannels, self.nJoints,1),
						)
					)
			else:
				_chantojoints.append(AttentionPartsCRF(self.nChannels, self.LRNSize, self.IterSize, self.nJoints))
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
		#print("1", x.mean())
		x = self.mp(x)
		x = self.res2(x)
		#print("2", x.mean())
		x = self.res3(x)
		out = []

		for i in range(self.nStack):
			#print("3", x.mean())
			x1 = self.hourglass[i](x)
			#print("4", x1.mean())
			x1 = self.Residual[i](x1)
			#print("5", x1.mean())
			x1 = self.lin1[i](x1)
			#print("6", x1.mean())
			out.append(self.chantojoints[i](x1))
			x1 = self.lin2[i](x1)
			#print("7", x1.mean())
			x = x + x1 + self.jointstochan[i](out[i])

		return (out)


# PoseAttention(nChannels, nStack, nModules, numReductions, nJoints, LRNSize, IterSize)
import numpy as np

if __name__ == "__main__":
    Net = PoseAttention(3, 4, 4, 1, 8, 1, 1)
    Net = Net.double()


    x  = np.random.randn(3, 3, 256,256)
    xt = torch.from_numpy(x)
    xt = xt.clone().detach()

    y = Net(xt)
    y = np.array(y)
    print(y.shape)
    print(y[0].shape)