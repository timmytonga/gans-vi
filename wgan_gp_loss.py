import torch


# gradient penalty: https://arxiv.org/pdf/1704.00028.pdf
def get_gradient(disc, real, fake, epsilon):
	"""
	Given a batch of real imgs (from dataset) and fake imgs (from generator) and epsilon
	Return the gradient of the critic's scores with respect to mixes of real and fake images.
		computes: grad(D(mix_img))
	"""
	# Mix the images together
	mixed_images = real * epsilon + fake * (1 - epsilon)

	# Calculate the critic's scores on the mixed images
	mixed_scores = disc(mixed_images)

	# Take the gradient of the scores with respect to the images
	gradient = torch.autograd.grad(
		inputs=mixed_images, outputs=mixed_scores,
		grad_outputs=torch.ones_like(mixed_scores),  # this keeps the device
		create_graph=True,
		retain_graph=True,
	)[0]
	return gradient


def gradient_penalty(gradient):
	"""
	Return the gradient penalty, given the gradients of a batch of images
	E[(norm(gradient) - 1)^2]
	"""
	# Flatten the gradients so that each row captures one image
	gradient = gradient.view(len(gradient), -1)
	# Calculate the magnitude of every row
	gradient_norm = gradient.norm(2, dim=1)
	# Penalize the mean squared distance of the gradient norms from 1
	penalty = torch.mean((gradient_norm - 1) ** 2)
	return penalty


# loss functions: in WGAN GP, we use the mean of each batch rather than BCE (original GAN formulation)
def gen_loss_wgan_gp(disc_output):
	""" Return the loss for the generator given the discriminator's output
	disc_output:  this is tensor of size (num_batch, 1) --> 2nd dim is score
	only depends on the discriminator's score (line 12 of algorithm 1 in WGAN GP paper)
	"""
	return -torch.mean(disc_output)


def disc_loss_gp(fake_scores, real_scores, grad_penal, c_lambda):
	""" Compute the discriminator's loss according to WGAN-GP i.e.
	E[D(G(z))] - E[D(x)] - lambda * grad_penal
	where given batch x and G(z):
	fake_scores = D(G(z)), real_scores = D(x)
	"""
	return torch.mean(fake_scores) - torch.mean(real_scores) + c_lambda * grad_penal


def disc_loss_wgan_gp(disc, real, fake, device):
	""" Combine everything above to simplify """
	fake_score = disc(fake.detach())  # no need to compute the grad here
	real_score = disc(real)
	epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)     # epsilon for GP
	gradient = get_gradient(disc, real, fake.detach(), epsilon)
	gp = gradient_penalty(gradient)
	return disc_loss_gp(fake_score, real_score, gp, c_lambda)
