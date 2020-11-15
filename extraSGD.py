# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:22:57 2020

@author: User
"""

import torch

from .optimizer import Optimizer, required

class extraSGD(Optimizer):
	
	def __init__(self, params, lr=required):
		if lr is not required and lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		defaults = dict(lr=lr)
		super(extraSGD, self).__init(params, defaults)
		self.params_copy = []
		
	
	@torch.no_grad()
	
	def extra(self, closure=None):
			
		for group in self.param_groups:
			for p in group['params']:
				self.params_copy.append(p.data.clone())	
				if p.grad is None:
					continue
				d_p = p.grad
				p.add_(d_p, alpha=-group['lr'])

		
	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()
		i = 0
		for group in self.param_groups:
			for p in group['params']:
				p.data = self.params_copy[i]
				i += 1
				if p.grad is None:
					continue
				d_p = p.grad
				p.add_(d_p, alpha=-group['lr'])
				
		self.params_copy = []		
		return loss