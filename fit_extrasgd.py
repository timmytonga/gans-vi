# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:13:28 2020

@author: Konstantina
"""

import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils import show_tensor_images
from collections import OrderedDict

def fit_extrasgd(gen, gen2, gen_opt, disc, disc2, disc_opt, device, n_epochs, trainset1, trainset2, disc_repeats, make_noise, z_dim, disc_loss_fn, gen_loss_fn, display_step, c_lambda, lr, checkpt=True, loadcheckptpath=None):
	# ================================================================== #
	#       Initialize training vars and load checkpt if any             #
	# ================================================================== #
	step = 0
	generator_losses = []
	disc_losses = []
	img_history = []
	last_epoch = 0
    #added for averaging
	disc_p = {}
	gen_p = {}
	disc_step = 0
	gen_step = 0
	gen2_losses = []
	disc2_losses = []



	if loadcheckptpath is not None:  # we were provided with checkpt. Load and resume
		print(f"========= Checkpoint found! Resuming training from last checkpoint at ======== \n{loadcheckptpath}")
		print("Loading checkpoint...")
		checkpoint = torch.load(loadcheckptpath)
		print("Checkpoint loaded successfully! Info:")
		last_epoch = checkpoint['epoch'] + 1
		print(f"\tLast epoch: {last_epoch - 1}")
		disc.load_state_dict(checkpoint['disc_state_dict'])
		gen.load_state_dict(checkpoint['gen_state_dict'])
		disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])
		gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
		print(f"\tLoaded discriminator and generator's old weight and state_dict")
		generator_losses = checkpoint['generator_losses']
		disc_losses = checkpoint['disc_losses']
		print(f"\tLoaded losses history.")
		step = checkpoint['step']
		print(f"\tLast step: {step}")
		# img_history = checkpoint['img_history']
		print("===================== FINISHED LOADING LAST CHECKPOINT. TRAINING ===========")

	# ==================== BEGIN TRAINING ==================== #
	print("Training with", device)
	for epoch in range(last_epoch, n_epochs):
		print(f"EPOCH: {epoch}")
		# Dataloader returns the batches
		for (real1, label1), (real2, label2) in tqdm(zip(trainset1, trainset2)):  # tqdm is just a progress bar
			# make sure they are training
			gen.train()
			disc.train()
			cur_batch_size = len(real1)
			real1 = real1.to(device)  # load to GPU if enabled
			real2 = real2.to(device)

			# ================================================================== #
			#                      Train the discriminator                       #
			# ================================================================== #
			mean_iteration_disc_loss = 0
			for _ in range(disc_repeats):  # train the discriminator this many times
				# ======== First compute the losses on the current model ======== #
				disc_opt.zero_grad()  # first zero out the grad

				# now we generate fake images and compute the loss based on the given loss fn
				noise = make_noise(cur_batch_size, z_dim, device=device)
				fake = gen(noise)
				# compute the loss (loss fn set above)
				disc_loss = disc_loss_fn(disc, real1, fake,device, c_lambda)
				# Keep track of the average discriminator loss in this batch
				mean_iteration_disc_loss += disc_loss.item() / disc_repeats

				# Update gradients
				disc_loss.backward(retain_graph=True)
				#keep previous weights
				with torch.no_grad():
				    for name, p in disc.named_parameters():
				      disc_p[name] = p
				# Update optimizer
				disc_opt.step()
                
				disc_opt.zero_grad()     
				noise = make_noise(cur_batch_size, z_dim, device=device)
				fake = gen(noise) 
				disc_loss = disc_loss_fn(disc, real2, fake, device, c_lambda)
                # Update gradients
				disc_loss.backward(retain_graph=True)
                # Update optimizer
				with torch.no_grad():
				  for name, p in disc.named_parameters():
				    disc_p[name] = disc_p[name]-lr*p.grad
				    new_state_dict = OrderedDict({name: disc_p[name]})
				    disc.load_state_dict(new_state_dict, strict=False)
				disc_p = {}

			disc_losses += [mean_iteration_disc_loss]
            
            #compute the loss for the averaged disc
			noise = make_noise(cur_batch_size, z_dim, device=device)
			fake = gen2(noise) 
			disc2_loss = disc_loss_fn(disc2, real1, fake, device, c_lambda)
			disc2_losses += [disc2_loss.item()]
			disc_step += 1
            
            #update the avg weights
			with torch.no_grad():
			    for name, p in disc.named_parameters():
			        q = disc2.state_dict()[name]
			        q = ((disc_step-1)*q +p)/disc_step 
			        new_state_dict = OrderedDict({name: q})
			        disc2.load_state_dict(new_state_dict, strict=False)
			# ================================================================== #
			#                      Train the generator                           #
			# ================================================================== #
			# first zero out the gradients
			gen_opt.zero_grad()
			# then generate some fake images
			morenoise = make_noise(cur_batch_size, z_dim, device=device)
			fake = gen(morenoise)
			# compute the scores and losses
			fake_score = disc(fake)
			gen_loss = gen_loss_fn(fake_score)
            # Keep track of the generator loss
			generator_losses += [gen_loss.item()]
			# backprop i.e. compute the gradients
			gen_loss.backward()
			with torch.no_grad():
				for name, p in gen.named_parameters():
				    gen_p[name] = p
			# Update the weights
			gen_opt.step()

			gen_opt.zero_grad()
			morenoise = make_noise(cur_batch_size, z_dim, device=device)
			fake = gen(morenoise)
			# compute the scores and losses
			fake_score = disc(fake)
			gen_loss = gen_loss_fn(fake_score)
			# backprop i.e. compute the gradients
			gen_loss.backward()
			# Update optimizer
			with torch.no_grad():
			    for name, p in gen.named_parameters():
			         gen_p[name] = gen_p[name]-lr*p.grad
			         new_state_dict = OrderedDict({name: gen_p[name]})
			         gen.load_state_dict(new_state_dict, strict=False)
			gen_p = {}
            
            # compute avg generator loss
			morenoise = make_noise(cur_batch_size, z_dim, device=device)
			fake = gen2(morenoise)
			fake_score = disc2(fake)
			gen2_loss = gen_loss_fn(fake_score)
			gen_step += 1
            
            #update the avg weights
			with torch.no_grad():
			    for name, p in gen.named_parameters():
			        q = gen2.state_dict()[name]
			        q = ((gen_step-1)*q +p)/gen_step 
			        gen2.state_dict()[name] = q
			        new_state_dict = OrderedDict({name: q})
			        gen2.load_state_dict(new_state_dict, strict=False)
        
        
			gen2_losses += [gen2_loss.item()]
			# ================================================================== #
			#                      Visualization                                 #
			# ================================================================== #
			if step % display_step == 0 and step > 0:
				gen_mean = sum(generator_losses[-display_step:]) / display_step
				disc_mean = sum(disc_losses[-display_step:]) / display_step
				print(f"Step {step}: Generator loss: {gen_mean}, Discriminator's loss: {disc_mean}")
				show_tensor_images(fake)
				show_tensor_images(real1)
				step_bins = 20
				num_examples = (len(generator_losses) // step_bins) * step_bins
				plt.plot(
					range(num_examples // step_bins),
					torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
					# range(len(generator_losses)),
					# generator_losses,
					label="Generator Loss"
				)
           

				plt.plot(
					range(num_examples // step_bins),
					torch.Tensor(disc_losses[:num_examples]).view(-1, step_bins).mean(1),
					# range(len(disc_losses)),
					# disc_losses,
					label="Discriminator Loss"
				)
				plt.title('Mean loss of batch')
				plt.legend()
				plt.show()

			step += 1

		# ================================================================== #
		#                       Validation (TODO)                            #
		# ================================================================== #
		print(f"================= VALIDATION for EPOCH {epoch}==================")
		print("-Overall losses:")
		## plot total losses after each epoch
		plt.plot(
			# range(num_examples // step_bins),
			# torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
			range(len(generator_losses)),
			generator_losses,
			label="Generator Loss"
		)
		plt.plot(
            # range(num_examples // step_bins),
            # torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
            range(len(gen2_losses)), 
            gen2_losses,
            label="Generator Avg Loss"
        )
		plt.plot(
			# range(num_examples // step_bins),
			# torch.Tensor(disc_losses[:num_examples]).view(-1, step_bins).mean(1),
			range(len(disc_losses)),
			disc_losses,
			label="Discriminator Loss"
		)
		plt.plot(
            # range(num_examples // step_bins), 
            # torch.Tensor(disc_losses[:num_examples]).view(-1, step_bins).mean(1),
            range(len(disc2_losses)),
            disc2_losses,
            label="Discriminator Avg Loss"
        )
		plt.ylabel('loss')
		plt.xlabel('number of batches processed')
		plt.title('Loss over epochs for discriminator and generator')
		plt.legend()
		plt.show()
		print("Images")
		valimg = gen(validation_noise)
		show_tensor_images(valimg)
		img_history += [valimg]

		# ================================================================== #
		#                      Check-pointing                                #
		# ================================================================== #

		if checkpt and epoch % check_point_after_epochs == 0 and epoch > 0:
			timestamp = get_cur_timestamp()
			checkpt_file_path = CHECKPOINT_PATH + timestamp + "checkpoint.pt"
			print(f"*** EPOCH {epoch} SAVING CHECKPOINTS AT {checkpt_file_path} ***")
			save_dict = {
				'epoch': epoch,
				'disc_state_dict': disc.state_dict(),
				'gen_state_dict': gen.state_dict(),
				'disc_opt_state_dict': disc_opt.state_dict(),
				'gen_opt_state_dict': gen_opt.state_dict(),
				'generator_losses': generator_losses,
				'disc_losses': disc_losses,
				'step': step,
				'img_history': img_history
			}
			torch.save(save_dict, checkpt_file_path)