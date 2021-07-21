# Code derived from https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/omd.py

import math
import torch
from torch.optim import Optimizer

required = object()


class AdaPEGAdamSVRG(Optimizer):
    def __init__(self, params, nbatches, model,
                 vr_from_epoch, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, squared_grad=False, optimistic=False, svrg=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, squared_grad=squared_grad, optimistic=optimistic, svrg=svrg)

        self.nbatches = nbatches
        self.model = model
        self.vr_from_epoch = vr_from_epoch
        self.batches_processed = 0
        self.epoch = 0
        self.running_tmp = {}
        self.svrg = svrg
        super(AdaPEGAdamSVRG, self).__init__(params, defaults)

    def initialize(self):
        for group in self.param_groups:
            for p in group['params']:
                gsize = p.data.size()
                gtbl_size = torch.Size([self.nbatches] + list(gsize))

                param_state = self.state[p]

                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        param_state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                if 'gktbl' not in param_state:
                    param_state['gktbl'] = torch.zeros(gtbl_size)

                if 'gavg' not in param_state:
                    param_state['gavg'] = p.data.clone().double().zero_()

    def store_old_table(self):
        """
        Stores the old gradient table for recalibration purposes.
        """
        if not self.svrg:
            return

        for group in self.param_groups:
            for p in group['params']:
                gk = p.grad.data

                param_state = self.state[p]

                gktbl = param_state['gktbl']
                gavg = param_state['gavg']

                param_state['gktbl_old'] = gktbl.clone()
                param_state['gavg_old'] = gavg.clone()

    def recalibrate_start(self):
        """ Part of the recalibration pass with SVRG.
        Stores the gradients for later use.
        """

        self.recalibration_i = 0
        self.initialize()
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

                    if self.svrg:
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

        variance = sum(variances)/len(variances)
        return variance

    def __setstate__(self, state):
        super(AdaPEGAdamSVRG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, batch_id, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.svrg:
                    param_state = self.state[p]
                    gktbl = param_state['gktbl']
                    gavg = param_state['gavg'].type_as(p.data)
                    gi = gktbl[batch_id, :].cuda()
                    p.grad.data.sub_(gi - gavg)
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaPEGAdam does not support sparse gradients, please consider SparseAdam instead')
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
