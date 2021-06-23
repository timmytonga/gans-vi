# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import print_function
import argparse
import pickle
import os
from timeit import default_timer as timer

from torch.autograd import Variable
import math
import datetime
import src.utils as utils

import logging
import pdb
from torch.nn.functional import nll_loss, log_softmax
import numpy as np

def recalibrate(model_params, train_loader, generator, discriminator, gen_optimizer, dis_optimizer, device):
    if hasattr(gen_optimizer, "recalibrate_start"):
        gen_optimizer.recalibrate_start(device=device)
    if hasattr(dis_optimizer, "recalibrate_start"):
        dis_optimizer.recalibrate_start(device=device)

    if gen_optimizer.vr_from_epoch is not None and gen_optimizer.epoch >= gen_optimizer.vr_from_epoch:
        for batch_idx, (x_true, target) in enumerate(train_loader):
            batch_id = batch_idx
            x_true = Variable(x_true)
            x_true = x_true.to(device=device)

            z = Variable(utils.sample(model_params["distribution"], (len(x_true), model_params["num_latent"])))
            z.to(device=device)

            x_gen = generator(z)
            for p in generator.parameters():
                p.requires_grad = False

            p_true, p_gen = discriminator(x_true), discriminator(x_gen)
            dis_loss = - utils.compute_gan_loss(p_true, p_gen, mode=model_params["mode"])
            if model_params["gradient_penalty"] != 0:
                penalty = discriminator.get_penalty(x_true.data, x_gen.data)
                dis_loss += penalty * model_params["gradient_penalty"]

            dis_optimizer.zero_grad()
            dis_loss.backward(retain_graph=True)
            closure = lambda: 0
            dis_optimizer.recalibrate(batch_id, closure)


            if model_params["mode"] == "wgan" and model_params["gradient_penalty"] == 0.0:
                for p in discriminator.parameters():
                    p.data.clamp_(-model_params["clip"], model_params["clip"])

            for p in generator.parameters():
                p.requires_grad = True

            for p in discriminator.parameters():
                p.requires_grad = False

            p_true, p_gen = discriminator(x_true), discriminator(x_gen)
            gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=model_params["mode"])

            gen_optimizer.zero_grad()
            gen_loss.backward(retain_graph=True)
            closure = lambda: 0
            gen_optimizer.recalibrate(batch_id, closure)

            for p in discriminator.parameters():
                p.requires_grad = True


    if hasattr(gen_optimizer, "recalibrate_end"):
        gen_optimizer.recalibrate_end()
    if hasattr(dis_optimizer, "recalibrate_end"):
        dis_optimizer.recalibrate_end()
