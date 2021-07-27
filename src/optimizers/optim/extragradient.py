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


class Extragradient(Optimizer):
    """Base class for optimizers with extrapolation step.

        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        super(Extragradient, self).__init__(params, defaults)
        self.params_copy = []
        m = self.defaults["nbatches"]
        for group in self.param_groups:
            for p in group['params']:
                gsize = p.data.size()
                gtbl_size = torch.Size([m] + list(gsize))

                param_state = self.state[p]

                if 'gktbl' not in param_state:
                    param_state['gktbl'] = torch.zeros(gtbl_size)

                if 'gavg' not in param_state:
                    param_state['gavg'] = p.data.clone().double().zero_()

    def update(self, p, group):
        raise NotImplementedError

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

    def extrapolation(self, batch_id):
        """Performs the extrapolation step and save a copy of the current parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if group["svrg"]:
                    gktbl = param_state['gktbl']
                    gavg = param_state['gavg'].type_as(p.data)
                    gi = gktbl[batch_id, :].cuda()
                    p.grad.data.sub_(gi - gavg)
                u = self.update(p, group)
                if is_empty:
                    # Save the current parameters for the update step. Several extrapolation step can be made before each update but only the parameters before the first extrapolation step are saved.
                    self.params_copy.append(p.data.clone())
                if u is None:
                    continue
                # Update the current parameters
                p.data.add_(u)

    def epoch_diagnostics(self):
        """
        Called after recalibrate, returns variance
        """
        m = self.defaults["nbatches"]

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
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.params_copy) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        loss = None
        if closure is not None:
            loss = closure()

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                if group["svrg"]:
                    gktbl = param_state['gktbl']
                    gavg = param_state['gavg'].type_as(p.data)
                    gi = gktbl[batch_id, :].cuda()
                    p.grad.data.sub_(gi - gavg)
                i += 1
                u = self.update(p, group)
                if u is None:
                    continue
                # Update the parameters saved during the extrapolation step
                p.data = self.params_copy[i].add_(u)

        # Free the old parameters
        self.params_copy = []
        return loss


class ExtraSGD(Extragradient):
    """Implements stochastic gradient descent with extrapolation step (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.ExtraSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.extrapolation()
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, svrg=True, nbatches=64):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, svrg=svrg, nbatches=nbatches)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ExtraSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def update(self, p, group):
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        if p.grad is None:
            return None
        d_p = p.grad.data
        if weight_decay != 0:
            d_p.add_(weight_decay, p.data)
        if momentum != 0:
            param_state = self.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf.mul_(momentum).add_(d_p)
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(1 - dampening, d_p)
            if nesterov:
                d_p = d_p.add(momentum, buf)
            else:
                d_p = buf

        return -group['lr'] * d_p


class ExtraAdam(Extragradient):
    """Implements the Adam algorithm with extrapolation step.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, svrg=True, nbatches=64):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, svrg=svrg, nbatches=nbatches)
        super(ExtraAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ExtraAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def update(self, p, group):
        if p.grad is None:
            return None
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if "step" not in state:
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

        return -step_size * exp_avg / denom
