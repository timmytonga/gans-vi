# import and show image function
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datetime import datetime     # to get time stamp for checkpts
import pytz


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


def get_cur_timestamp():
    tz_NY = pytz.timezone('America/New_York')
    dateTimeObj = datetime.now(tz_NY)
    return dateTimeObj.strftime("[%d-%b-%Y(%H:%M)]")

