# Code derived from https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/omd.py

import math
import torch
from torch.optim import Optimizer

required = object()


class AdaPEGAdamSVRG(Optimizer):
    def __init__(self, params, nbatches, model, vr_bn_at_recalibration,
                 vr_from_epoch, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, squared_grad=False, optimistic=False, batchnormreset=False):
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

        self.nbatches = nbatches
        self.model = model
        self.vr_bn_at_recalibration = vr_bn_at_recalibration
        self.vr_from_epoch = vr_from_epoch
        self.batches_processed = 0
        self.epoch = 0
        self.running_tmp = {}
        self.batchnormreset = batchnormreset
        super(AdaPEGAdamSVRG, self).__init__(params, defaults)

    def initialize(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]

                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    amsgrad = group['amsgrad']
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        param_state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                if 'gavg' not in param_state:
                    param_state['gavg'] =  p.data.double().clone().zero_()
                    param_state['gi'] = p.data.clone().zero_()
                    param_state['gi_debug'] = p.data.clone().zero_()

                if 'tilde_x' not in param_state:
                    param_state['tilde_x'] = p.data.clone()
                    param_state['xk'] = p.data.clone()

        # Batch norm's activation running_mean/var variables
        state = self.model.state_dict()
        for skey in state.keys():
            if skey.endswith(".running_mean") or skey.endswith(".running_var"):
                self.running_tmp[skey] = state[skey].clone()

    def recalibrate_start(self):
        """ Part of the recalibration pass with SVRG.
        Stores the gradients for later use.
        """
        self.epoch += 1
        self.recal_calls = 0
        self.initialize()
        if self.batchnormreset:
            self.store_running_mean()
        print("Recal epoch: {}".format(self.epoch))

        if self.epoch >= self.vr_from_epoch:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    gavg = param_state['gavg']
                    gavg.zero_()

                    tilde_x = param_state['tilde_x']
                    tilde_x.zero_().add_(p.data.clone())
                    #pdb.set_trace()


    def recalibrate(self, batch_id, closure):
        """ Compute part of the full batch gradient, from one minibatch
        """
        loss = closure()

        if self.epoch >= self.vr_from_epoch:
            self.recal_calls += 1
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    gk = p.grad.data

                    param_state = self.state[p]

                    gavg = param_state['gavg']
                    gavg.add_(1.0/self.nbatches, gk.double())

        return loss

    def recalibrate_end(self):
        if self.batchnormreset:
            self.restore_running_mean()
        if self.recal_calls != self.nbatches:
            raise Exception("recalibrate_end called, with {} nbatches: {}".format(
                            self.recal_calls, self.nbatches))

    def store_running_mean(self):
        # Store running_mean/var temporarily
        state = self.model.state_dict()
        #pdb.set_trace()
        for skey in self.running_tmp.keys():
            self.running_tmp[skey].zero_().add_(state[skey])

    def restore_running_mean(self):
        state = self.model.state_dict()
        for skey in self.running_tmp.keys():
            state[skey].zero_().add_(self.running_tmp[skey])

    def __setstate__(self, state):
        super(AdaPEGAdamSVRG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, batch_id, closure=None):


        if self.epoch >= self.vr_from_epoch:
            if self.batchnormreset:
                self.store_running_mean()
            ## Store current xk, replace with x_tilde
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    xk = param_state['xk']
                    xk.zero_().add_(p.data)
                    p.data.zero_().add_(param_state['tilde_x'])

            # Standard is vr_bn_at_recalibration=True, so this doesn't fire.
            if not self.vr_bn_at_recalibration:
                self.model.eval() # turn off batch norm
            ## Compute gradient at x_tilde
            closure()

            ## Store x_tilde gradient in gi, and revert to xk
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    xk = param_state['xk']
                    gi = param_state['gi']
                    gi.zero_().add_(p.grad.data)
                    p.data.zero_().add_(xk)

            # Make sure batchnorm is handled correctly.
            if self.batchnormreset:
                self.restore_running_mean()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                gk = p.grad.data
                if gk.is_sparse:
                    raise RuntimeError(
                        'AdaPEGAdam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                param_state = self.state[p]

                gi = param_state['gi']
                gavg = param_state['gavg']

                if self.epoch >= self.vr_from_epoch:
                    grad = gk.clone().sub_(gi).add_(gavg.type_as(gk))
                else:
                    grad = gk.clone()  # Just do sgd steps

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
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) #OptimisticAdam
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

                p.data.addcdiv_(-2 * step_size, exp_avg, denom)

                if state['step'] > 1:
                    if group['optimistic']:
                        p.data.addcdiv_(group['lr'], state['exp_avg_previous'],
                                        state['exp_avg_sq_previous'])  # OptimisticAdam
                    else:
                        p.data.addcdiv_(group['lr'], state['exp_avg_previous'], denom)

                state['exp_avg_previous'] = exp_avg.clone() / (bias_correction1)
                if group['optimistic']:
                    state['exp_avg_sq_previous'] = denom.clone() / math.sqrt(bias_correction2)  # OptimisticAdam
                state['grad_prev'] = grad.clone()

        return loss
