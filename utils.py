# import and show image function
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
# torchvision
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def show_tensor_images(image_tensor, num_images=25):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.

    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def make_noise(num_noise, z_dim=128, device='cpu'):
    """ Returns a tensor of size (num_noise, z_dim)"""
    return torch.randn(num_noise, z_dim, device=device)

def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
	if isinstance(m, nn.BatchNorm2d):
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
		torch.nn.init.constant_(m.bias, 0)
        
def weights_init_zero(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
      torch.nn.init.constant_(m.weight, 0.0)
  if isinstance(m, nn.BatchNorm2d):
      torch.nn.init.constant_(m.weight, 0.0)
      torch.nn.init.constant_(m.bias, 0.0)
        
def get_cur_timestamp():
	tz_NY = pytz.timezone('America/New_York')
	dateTimeObj = datetime.now(tz_NY)
	return dateTimeObj.strftime("[%d-%b-%Y(%H:%M)]")