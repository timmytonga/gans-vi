# Code derived from https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/omd.py

import math
import torch
from torch.optim import Optimizer

required = object()

class AdaPEGAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, squared_grad = False, optimistic = False):
        if not 0.0 <= lr:
         raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
         raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
         raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
         raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                     weight_decay=weight_decay, amsgrad=amsgrad, squared_grad=squared_grad, optimistic=optimistic)
        super(AdaPEGAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaPEGAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    return None
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaPEGAdam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                #exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) #OptimisticAdam
                if state['step'] == 1 or group['squared_grad']:
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad - state['grad_prev'], grad - state['grad_prev'])

           
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
                    if group['optimistic']:
                        p.data.addcdiv_(group['lr'], state['exp_avg_previous'], state['exp_avg_sq_previous']) #OptimisticAdam
                    else:
                        p.data.addcdiv_(group['lr'], state['exp_avg_previous'], denom)

                state['exp_avg_previous'] = exp_avg.clone()/(bias_correction1)
                if group['optimistic']:
                    state['exp_avg_sq_previous'] = denom.clone()/math.sqrt(bias_correction2) #OptimisticAdam
                state['grad_prev'] = grad.clone()

        return loss
    