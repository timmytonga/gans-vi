import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

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