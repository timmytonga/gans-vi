import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10    # torchvision
import torchvision.transforms as transforms

from dcgan_models import Generator, Discriminator
from wgan_gp_loss import disc_loss_wgan_gp, gen_loss_wgan_gp
from utils import make_noise, show_tensor_images, get_cur_timestamp, weights_init, weights_init_zero
from inception_score import inception_score

from tqdm.auto import tqdm  # for visualizing progress per epoch
import time


# =========================================================================== #
#                  FIRST, MUST MAKE SURE PATHS ARE CORRECT                    #
# =========================================================================== #
CHECKPOINT_PATH="/content/drive/My Drive/cs6140/gan_train_checkpt/"
SAVE_MODEL_PATH="/content/drive/My Drive/cs6140/gan_model/"
DISC_PATH = SAVE_MODEL_PATH + 'disc_weight.pth'
GEN_PATH = SAVE_MODEL_PATH + 'gen_weight.pth'


# =========================================================================== #
#                  SET ANY CHECKPOINT FILE HERE (ignore if no checkpoint)     #
# =========================================================================== #
CHECKPOINTFILENAME = "[12-Nov-2020(00:34)]checkpoint.pt"
CHECKPOINTFILEPATH = CHECKPOINT_PATH + CHECKPOINTFILENAME  # change the last part
ENABLE_CHECKPOINT = True


# =========================================================================== #
#                  LOAD MODEL IF AVAILABLE                                    #
# =========================================================================== #
wgan_gp = True    # do we want to do WGAN or WGAN-GP?

device = 'cpu'
if torch.cuda.is_available():  # use gpu if available
	device = 'cuda'


def load_model(dev):
	_gen = Generator()
	_disc = Discriminator()
	_disc.load_state_dict(torch.load(DISC_PATH))
	_gen.load_state_dict(torch.load(GEN_PATH))
	return _gen.to(dev), _disc.to(dev)


# initialize the models
gen = Generator().to(device)
disc = Discriminator(wgan_gp=wgan_gp).to(device)
# gen2 and disc2 are for averaging
gen2 = Generator().to(device)
disc2 = Discriminator(wgan_gp=wgan_gp).to(device)
# UNCOMMENT/COMMENT BELOW FOR LOADING MODEL
# gen, disc = load_model(device)

# =========================================================================== #
#                   HYPERPARAMETERS AND TRAINING DATA                         #
# =========================================================================== #
z_dim = 128
lr = 0.00002      # learning rate
if wgan_gp:       # according to varineq paper
	lr = 0.0001
beta_1 = 0.5      # first moment's momentum (ADAM hyperparam)
beta_2 = 0.9      # second moment's momentum
weight_clipping = 0.01
c_lambda = 10     # for gradient penalty
batch_size = 64

# get data and loader
transform_train = transforms.Compose([
	# transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = DataLoader(CIFAR10(root='.', download=True, transform=transform_train), batch_size=batch_size, shuffle=True)
trainset2 = DataLoader(CIFAR10(root='.', download=True, transform=transform_train), batch_size=batch_size, shuffle=True)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
gen2 = gen2.apply(weights_init_zero)
disc2 = disc2.apply(weights_init_zero)

# =========================================================================== #
#                  SET OPTIMIZERS HERE                                        #
# =========================================================================== #
# define the Adam optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))


# =========================================================================== #
#                  SET LOSS FUNCTIONS FOR FITTING                             #
# =========================================================================== #
disc_loss_fn = disc_loss_wgan_gp
gen_loss_fn = gen_loss_wgan_gp

# =========================================================================== #
#                  SET VALIDATION STUFF                                       #
# =========================================================================== #
torch.manual_seed(0)
validation_noise = make_noise(25, z_dim, device=device)  # so we have a fixed validation noise
torch.manual_seed(int(time.time()))  # this should be random enough for fitting


import copy

# =========================================================================== #
#                  FITTING                                                    #
# =========================================================================== #
display_step = 500  # for displaying info after this many step (i.e. batch)
disc_repeats = 5  # how many times to train the discriminator per one generator
check_point_after_epochs = 10  # checkpointing every these epochs
n_epochs = 100


def fit(checkpt=True, loadcheckptpath=None, checkpoint_name_tag="avgextraAdam.pt"):
	""" fit the models gen, disc in global var
	CHECKPOINT_NAME_TAG specifies the naming for the checkpoint file
	"""
	# ================================================================== #
	#       Initialize training vars                                     #
	# ================================================================== #
	step = 0
	generator_losses = []
	disc_losses = []
	inception_scores = []
	avgmodel_inception_scores = []
	img_history = []
	avgmodel_image_history = []
	last_epoch = 0
	gen2_losses = []
	disc2_losses = []

	# ================================================================== #
	#               Load checkpts if any                                 #
	# ================================================================== #
	if loadcheckptpath is not None:  # we were provided with checkpt. Load and resume
		print(f"========= Checkpoint found! Resuming training from last checkpoint at ======== \n{loadcheckptpath}")
		print("Loading checkpoint...")
		checkpoint = torch.load(loadcheckptpath)
		print("Checkpoint loaded successfully! Info:")
		last_epoch = checkpoint['epoch'] + 1
		print(f"\tLast epoch: {last_epoch - 1}")
		disc.load_state_dict(checkpoint['disc_state_dict'])
		gen.load_state_dict(checkpoint['gen_state_dict'])
		disc2.load_state_dict(checkpoint['disc2_state_dict'])
		gen2.load_state_dict(checkpoint['gen2_state_dict'])
		disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])
		gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
		print(f"\tLoaded discriminator and generator's old weight and state_dict")
		generator_losses = checkpoint['generator_losses']
		disc_losses = checkpoint['disc_losses']
		gen2_losses = checkpoint['gen2_losses']
		disc2_losses = checkpoint['disc2_losses']
		print(f"\tLoaded losses history.")
		step = checkpoint['step']
		print(f"\tLast step: {step}")
		img_history = checkpoint['img_history']
		inception_scores = checkpoint['inception_scores']
		avgmodel_image_history = checkpoint['avgmodel_image_history']
		avgmodel_inception_scores = checkpoint['avgmodel_inception_scores']
		print("Loaded img_history and inception_scores")
		print("===================== FINISHED LOADING LAST CHECKPOINT. TRAINING ===========")
	else:
		print("Checkpoint not found. Below is the first image with the validation noise.")
		firstimg = gen(validation_noise)
		show_tensor_images(firstimg)
		img_history += [firstimg]

	# ==================== BEGIN TRAINING ==================== #
	print("Training with", device)
	for epoch in range(last_epoch, n_epochs):
		# make sure the models are set to training
		gen.train()
		disc.train()
		print(f"EPOCH: {epoch}")
		# Dataloader returns the batches
		# Here, we sample two batches: one for extrapolation and one for gradient computation
		for (real1, label1), (real2, label2) in tqdm(zip(trainset, trainset2), total=len(trainset)):
			# tqdm is just a progress bar
			cur_batch_size = len(real1)
			real1 = real1.to(device)  # load to GPU if enabled
			real2 = real2.to(device)

			# ================================================================== #
			#                      EXTRAPOLATED STEP                             #
			# ================================================================== #
			# We save the original weights for later update
			disc_saved_state = copy.deepcopy(disc.state_dict())
			discopt_saved_state = copy.deepcopy(disc_opt.state_dict())
			gen_saved_state = copy.deepcopy(gen.state_dict())
			genopt_saved_state = copy.deepcopy(gen_opt.state_dict())

			# ===== First, we compute the extrapolation weights to compute the gradients on using the curr. weights ====
			# first zero out the grad
			disc_opt.zero_grad()
			# now compute the loss
			# first, we generate fake images and compute the loss based on the given loss fn
			noise = make_noise(cur_batch_size, z_dim, device=device)
			fake1 = gen(noise)
			# compute the loss (loss fn set above)
			disc_loss = disc_loss_fn(disc, real1, fake1, device, c_lambda)
			# Get gradients
			disc_loss.backward(retain_graph=True)

			# Load the extrapolated weights into our discriminator
			disc_opt.step()

			# ===== now, we do the same for the generator =====
			# first zero out the gradients
			gen_opt.zero_grad()
			# then generate some fake images
			morenoise = make_noise(cur_batch_size, z_dim, device=device)
			fake2 = gen(morenoise)
			# compute the scores and losses
			fake_score = disc(fake2)
			gen_loss = gen_loss_fn(fake_score)
			# compute the gradients
			gen_loss.backward()

			# Load the extrapolated weights into our discriminator generator. Must recover later with the saved states
			gen_opt.step()

			# ================================================================== #
			#       Obtain the extrapolated gradients for discriminator          #
			# ================================================================== #
			# we do it again but with the second mini batch and the extrapolated weights now
			# first zero out the grad
			disc_opt.zero_grad()
			# generate fake images again and compute the loss based on the given loss fn
			noise = make_noise(cur_batch_size, z_dim, device=device)
			fake = gen(noise)
			# compute the loss on the second batch (real2) and the extrapolated weight now
			disc_loss = disc_loss_fn(disc, real2, fake, device, c_lambda)
			# compute the gradient
			disc_loss.backward(retain_graph=True)
			# now the gradients is stored in the .grad of the parameters in this extrapolated disc
			# we set the just the weights back to before to optimize
			disc_opt.load_state_dict(discopt_saved_state)  # more sneaky move
			for name, p in disc.named_parameters():
				p.data = disc_saved_state[name].data  # sneaky move
			# add the loss to loss history
			disc_losses += [disc_loss.item()]
			disc_opt.step()

			# ================================================================== #
			#       Obtain the extrapolated gradients for generator              #
			# ================================================================== #
			# first zero out the gradients
			gen_opt.zero_grad()
			# then generate some fake images
			morenoise = make_noise(cur_batch_size, z_dim, device=device)
			fake = gen(morenoise)
			# compute the scores and losses
			fake_score = disc(fake)
			gen_loss = gen_loss_fn(fake_score)
			# compute the gradients on the extrapolated weights
			gen_loss.backward()
			# now the gradients is stored in the .grad of the parameters in this extrapolated disc
			# we set the just the weights back to before to optimize
			gen_opt.load_state_dict(genopt_saved_state)  # more sneaky move
			for name, p in gen.named_parameters():
				p.data = gen_saved_state[name].data  # sneaky move
			# Keep track of the generator loss
			generator_losses += [gen_loss.item()]
			gen_opt.step()  # sneaky move gets run here

			step += 1

			# ================================================================== #
			#                               Average                              #
			# ================================================================== #
			# compute the loss for the averaged disc
			noise = make_noise(cur_batch_size, z_dim, device=device)
			fake = gen2(noise)
			disc2_loss = disc_loss_fn(disc2, real1, fake, device, c_lambda)
			disc2_losses += [disc2_loss.item()]
			fake_score = disc2(fake)
			gen2_loss = gen_loss_fn(fake_score)
			gen2_losses += [gen2_loss.item()]

			# update the avg weights
			with torch.no_grad():
				for name, p in disc.named_parameters():
					q = disc2.state_dict()[name]
					q = ((step - 1) * q + p) / step
					new_state_dict = {name: q}
					disc2.load_state_dict(new_state_dict, strict=False)

			with torch.no_grad():
				for name, p in gen.named_parameters():
					q = gen2.state_dict()[name]
					q = ((step - 1) * q + p) / step
					new_state_dict = {name: q}
					gen2.load_state_dict(new_state_dict, strict=False)

		# ================================================================== #
		#                       Validation                                   #
		# ================================================================== #
		gen.eval()
		disc.eval()
		print(f"================= VALIDATION for EPOCH {epoch}==================")
		print("-Overall losses:")
		# plot total losses after each epoch
		plt.plot(
			# range(num_examples // step_bins),
			# torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
			range(len(generator_losses)),
			generator_losses,
			label="Generator Loss"
		)
		plt.plot(
			# range(num_examples // step_bins),
			# torch.Tensor(disc_losses[:num_examples]).view(-1, step_bins).mean(1),
			range(len(disc_losses)),
			disc_losses,
			label="Discriminator Loss"
		)
		# plot the avg weight models' losses
		plt.plot(
			# range(num_examples // step_bins),
			# torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
			range(len(gen2_losses)),
			gen2_losses,
			label="Avg Weight Generator Loss"
		)
		plt.plot(
			# range(num_examples // step_bins),
			# torch.Tensor(disc_losses[:num_examples]).view(-1, step_bins).mean(1),
			range(len(disc2_losses)),
			disc2_losses,
			label="Avg Weight Discriminator Loss"
		)
		plt.ylabel('loss')
		plt.xlabel('number of batches processed')
		plt.title('Loss over numbatches for discriminator and generator')
		plt.legend()
		plt.show()

		print("Showing Validation Images")
		valimg = gen(validation_noise)
		avgmodel_valimg = gen2(validation_noise)

		show_tensor_images(valimg)

		img_history += [valimg]
		avgmodel_image_history += [avgmodel_valimg]
		# print("Showing Images Generated by Random Input")
		# show_tensor_images(gen(make_noise(25, z_dim, device=device)))

		# === We validate our progress by measuring inception score on "validation noise" that we fixed in the beginning =====
		incept_score = inception_score(valimg, cuda=True, batch_size=batch_size, resize=True, n_sec=1)[0]
		avgmodel_incept_score = inception_score(avgmodel_valimg, cuda=True, batch_size=batch_size, resize=True, n_sec=1)[0]
		inception_scores.append(incept_score)
		avgmodel_inception_scores.append(avgmodel_incept_score)
		plt.plot(
			range(len(inception_scores)),
			inception_scores,
			label="No Averaging"
		)
		plt.plot(
			range(len(avgmodel_inception_scores)),
			avgmodel_inception_scores,
			label="With Averaging"
		)
		plt.xlabel("Epoch")
		plt.ylabel("Inception Score")
		plt.title("Inception Scores over Epochs")
		plt.legend()
		plt.show()

		# ================================================================== #
		#                      Check-pointing                                #
		# ================================================================== #

		if checkpt and epoch % check_point_after_epochs == 0 and epoch > 0:
			timestamp = get_cur_timestamp()
			checkpt_file_path = CHECKPOINT_PATH + timestamp + checkpoint_name_tag
			print(f"*** EPOCH {epoch} SAVING CHECKPOINTS AT {checkpt_file_path} ***")
			save_dict = {
				'epoch': epoch,
				'disc_state_dict': disc.state_dict(),
				'gen_state_dict': gen.state_dict(),
				'disc2_state_dict': disc2.state_dict(),
				'gen2_state_dict': gen2.state_dict(),
				'disc_opt_state_dict': disc_opt.state_dict(),
				'gen_opt_state_dict': gen_opt.state_dict(),
				'generator_losses': generator_losses,
				'disc_losses': disc_losses,
				'gen2_losses': gen2_losses,
				'disc2_losses': disc2_losses,
				'step': step,
				'img_history': img_history,
				'inception_scores': inception_scores,
				'avgmodel_image_history': avgmodel_image_history,
				'avgmodel_inception_scores': avgmodel_inception_scores
			}
			torch.save(save_dict, checkpt_file_path)


# FIT WITHOUT CHECKPOINTS ####
# fit()

# USE THIS FIT VERSION FOR FITTING WITH CHECKPOINTS ###
fit(checkpt=ENABLE_CHECKPOINT, loadcheckptpath=CHECKPOINTFILEPATH)


# ================= #
#  SAVE MODEL...    #
# ================= #
def save_model():
	torch.save(disc.state_dict(), SAVE_MODEL_PATH + 'disc_weight.pth')
	torch.save(gen.state_dict(), SAVE_MODEL_PATH + 'gen_weight.pth')
	torch.save(disc2.state_dict(), SAVE_MODEL_PATH + 'disc2_weight.pth')
	torch.save(gen2.state_dict(), SAVE_MODEL_PATH + 'gen2_weight.pth')


save_model()
