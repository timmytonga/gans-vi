
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
from torchvision.utils import make_grid
import torch
irange = range

def clip_weights(params, clip=0.01):
    for p in params:
        p.clamp_(-clip, clip)

def unormalize(x):
    return x/2. + 0.5

def sample(name, size):
    if name == 'normal':
        return torch.zeros(size).normal_()
    elif name == 'uniform':
        return torch.zeros(size).uniform_()
    else:
        raise ValueError()

def weight_init(m, mode='normal'):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        if mode == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'kaimingu':
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.)
        elif mode == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, 0.8)

def compute_gan_loss(p_true, p_gen, mode='gan', gen_flag=False):
    if mode == 'ns-gan' and gen_flag:
        loss = (p_true.clamp(max=0) - torch.log(1+torch.exp(-p_true.abs()))).mean() - (p_gen.clamp(max=0) - torch.log(1+torch.exp(-p_gen.abs()))).mean()
    elif mode == 'gan' or mode == 'gan++':
        loss = (p_true.clamp(max=0) - torch.log(1+torch.exp(-p_true.abs()))).mean() - (p_gen.clamp(min=0) + torch.log(1+torch.exp(-p_gen.abs()))).mean()
    elif mode == 'wgan':
        loss = p_true.mean() - p_gen.mean()
    else:
        raise NotImplementedError()

    return loss


def image_data(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
) -> None:
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im
