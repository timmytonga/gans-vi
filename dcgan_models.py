# import and show image function
import torch
from torch import nn


# now we write the generator and discriminator for DC GAN

class Generator(nn.Module):
	"""
	Generator for DCGAN.
		z_dim: dimension of noise vector
	"""

	def __init__(self, z_dim=128):
		super(Generator, self).__init__()
		self.z_dim = z_dim
		self.layer1 = nn.Sequential(
			nn.ConvTranspose2d(self.z_dim, 512, kernel_size=4),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True)
		)
		self.layer2 = nn.Sequential(
			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True)
		)
		self.layer3 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True)
		)
		self.final_layer = nn.Sequential(
			nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
		)

	def forward(self, noise):
		# first transform z to the appropriate dimension
		z = noise.view(len(noise), self.z_dim, 1, 1)
		# apply each layer
		z = self.layer1(z)
		z = self.layer2(z)
		z = self.layer3(z)
		z = self.final_layer(z)
		return z


# discriminator
class Discriminator(nn.Module):
	"""
	Discriminator for DCGAN
		wgan_gp (bool): specify whether to include batchnorm
	"""

	def __init__(self, wgan_gp=False):
		super(Discriminator, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(negative_slope=0.2)
		)

		if wgan_gp:  # no batchnorm for wgan_gp
			self.layer2 = nn.Sequential(
				nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
				nn.LeakyReLU(negative_slope=0.2)
			)
			self.layer3 = nn.Sequential(
				nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
				nn.LeakyReLU(negative_slope=0.2)
			)

		self.final_layer = nn.Linear(256 * 4 * 4, 1)  # just a linear layer for wgan

	def forward(self, img):
		x = self.layer1(img)
		# print("first layer size: ", x.shape)
		x = self.layer2(x)
		# print("second layer size: ", x.shape)
		x = self.layer3(x)
		# flatten
		x = x.view(x.size(0), -1)
		x = self.final_layer(x)
		return x


def make_noise(num_noise, z_dim=128, device='cpu'):
	""" Returns a tensor of size (num_noise, z_dim)"""
	return torch.randn(num_noise, z_dim, device=device)