#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

import math
import torch
from torch.optim import Optimizer

required = object()


class OMD(Optimizer):
    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(OMD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OMD, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['previous_update'] = torch.zeros_like(d_p)

                p.data.add_(-2*group['lr'], d_p).add_(group['lr']*state['previous_update'])

                state['previous_update'] = d_p

        return loss


class OptimisticAdam(Optimizer):
    def __init__(self, params, nbatches, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, svrg=False):
        if not 0.0 <= lr:
         raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
         raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
         raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
         raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                     weight_decay=weight_decay, amsgrad=amsgrad, svrg=svrg)
        super(OptimisticAdam, self).__init__(params, defaults)

        self.params_copy = []
        m = nbatches
        self.nbatches = nbatches
        for group in self.param_groups:
            for p in group['params']:
                gsize = p.data.size()
                gtbl_size = torch.Size([m] + list(gsize))

                param_state = self.state[p]

                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        param_state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                if 'gktbl' not in param_state:
                    param_state['gktbl'] = torch.zeros(gtbl_size)

                if 'gavg' not in param_state:
                    param_state['gavg'] = p.data.clone().double().zero_()

    def __setstate__(self, state):
        super(OptimisticAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def recalibrate_start(self):
        """ Part of the recalibration pass with SVRG.
        Stores the gradients for later use.
        """

        self.recalibration_i = 0

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['gavg'].zero_()

                # xk is changed to the running_x
                # p.data.zero_().add_(param_state['running_x'])
                # param_state['tilde_x'] = p.data.clone()

    def recalibrate(self, batch_id, closure):
        """ Part of the recalibration pass with SVRG.
        Stores the gradients for later use.
        """
        loss = closure()
        # print("recal loss:", loss)

        self.recalibration_i += 1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                gk = p.grad.data.double()

                param_state = self.state[p]

                gktbl = param_state['gktbl']
                gavg = param_state['gavg']

                # pdb.set_trace()

                # Online mean/variance calcuation from wikipedia
                delta = gk - gavg
                gavg.add_(1.0 / self.recalibration_i, delta)

                #########
                gktbl[batch_id, :] = p.grad.data.cpu().clone()

        return loss

    def store_old_table(self):
        """
        Stores the old gradient table for recalibration purposes.
        """
        if not self.defaults["svrg"]:
            return

        for group in self.param_groups:
            for p in group['params']:

                param_state = self.state[p]

                gktbl = param_state['gktbl']
                gavg = param_state['gavg']

                param_state['gktbl_old'] = gktbl.clone()
                param_state['gavg_old'] = gavg.clone()

    def epoch_diagnostics(self):
        """
        Called after recalibrate, returns variance
        """
        m = self.nbatches

        layernum = 0
        layer_gradient_norm_sqs = []
        gavg_norm_acum = 0.0
        gavg_acum = []
        for group in self.param_groups:
            for p in group['params']:

                layer_gradient_norm_sqs.append([])
                gavg = self.state[p]['gavg'].cpu()
                gavg_acum.append(gavg.numpy())
                gavg_norm_acum += gavg.norm()**2 #torch.dot(gavg, gavg)
                layernum += 1

        gradient_norm_sqs = []
        vr_step_variance = []
        cos_acums = []
        variances = []

        for batch_id in range(m):
            norm_acum = 0.0
            ginorm_acum = 0.0
            vr_acum = 0.0
            layernum = 0
            cos_acum = 0.0
            var_acum = 0.0
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]

                    gktbl = param_state['gktbl']
                    gavg = param_state['gavg'].type_as(p.data).cpu()

                    gi = gktbl[batch_id, :]
                    var_norm_sq = (gi-gavg).norm()**2 #torch.dot(gi-gavg, gi-gavg)
                    norm_acum += var_norm_sq
                    ginorm_acum += gi.norm()**2 #torch.dot(gi, gi)
                    layer_gradient_norm_sqs[layernum].append(var_norm_sq)

                    if group["svrg"]:
                        gktbl_old = param_state['gktbl_old']
                        gavg_old = param_state['gavg_old'].type_as(p.data).cpu()
                        gi_old = gktbl_old[batch_id, :]
                        #pdb.set_trace()
                        vr_step = gi - gi_old + gavg_old
                    else:
                        vr_step = gi
                    vr_acum += (vr_step - gavg).norm()**2 #torch.dot(vr_step - gavg, vr_step - gavg)
                    cos_acum += torch.sum(gavg*gi)

                    var_acum += (gi - gavg).norm()**2

                    layernum += 1
            gradient_norm_sqs.append(norm_acum)
            vr_step_variance.append(vr_acum)
            cosim = cos_acum/math.sqrt(ginorm_acum*gavg_norm_acum)
            #pdb.set_trace()
            cos_acums.append(cosim)
            variances.append(var_acum)

        if self.defaults["svrg"]:
            variance = sum(vr_step_variance)/len(vr_step_variance)
        else:
            variance = sum(variances)/len(variances)
        return variance

    def step(self, batch_id, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    return None

                if group["svrg"]:
                    param_state = self.state[p]
                    gktbl = param_state['gktbl']
                    gavg = param_state['gavg'].type_as(p.data)
                    gi = gktbl[batch_id, :].cuda()
                    p.grad.data.sub_(gi - gavg)

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-2*step_size, exp_avg, denom)

                if state['step'] > 1:
                    p.data.addcdiv_(group['lr'], state['exp_avg_previous'], state['exp_avg_sq_previous'])

                state['exp_avg_previous'] = exp_avg.clone()/(bias_correction1)
                state['exp_avg_sq_previous'] = denom.clone()/math.sqrt(bias_correction2)

        return loss
