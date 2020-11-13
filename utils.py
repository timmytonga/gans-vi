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

