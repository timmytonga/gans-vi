import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
# torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from dcgan_models import Generator, Discriminator
from wgan_gp_loss import disc_loss_wgan_gp, gen_loss_wgan_gp
from utils import make_noise, show_tensor_images, get_cur_timestamp


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


# before training initialize weight
def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
	if isinstance(m, nn.BatchNorm2d):
		torch.nn.init.normal_(m.weight, 0.0, 0.02)
		torch.nn.init.constant_(m.bias, 0)


gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


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
#                  FITTING                                                    #
# =========================================================================== #
display_step = 1000  # for displaying info after this many step (i.e. batch)
disc_repeats = 5  # how many times to train the discriminator per one generator
check_point_after_epochs = 10  # checkpointing every these epochs
n_epochs = 100
save_dict = None  # backup checkpoint for manual saving
torch.manual_seed(0)
validation_noise = make_noise(25, z_dim, device=device)  # so we have a fixed validation noise
torch.manual_seed(int(time.time()))  # this should be random enough for fitting


def fit(checkpt=True, loadcheckptpath=None):
	# ================================================================== #
	#       Initialize training vars and load checkpt if any             #
	# ================================================================== #
	step = 0
	generator_losses = []
	disc_losses = []
	img_history = []
	last_epoch = 0

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
		for real, label in tqdm(trainset):  # tqdm is just a progress bar
			# make sure they are training
			gen.train()
			disc.train()
			cur_batch_size = len(real)
			real = real.to(device)  # load to GPU if enabled

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
				disc_loss = disc_loss_fn(disc, real, fake)
				# Keep track of the average discriminator loss in this batch
				mean_iteration_disc_loss += disc_loss.item() / disc_repeats

				# Update gradients
				disc_loss.backward(retain_graph=True)
				# Update optimizer
				disc_opt.step()

			disc_losses += [mean_iteration_disc_loss]

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
			# backprop i.e. compute the gradients
			gen_loss.backward()
			# Update the weights
			gen_opt.step()

			# Keep track of the generator loss
			generator_losses += [gen_loss.item()]

			# ================================================================== #
			#                      Visualization                                 #
			# ================================================================== #
			if step % display_step == 0 and step > 0:
				gen_mean = sum(generator_losses[-display_step:]) / display_step
				disc_mean = sum(disc_losses[-display_step:]) / display_step
				print(f"Step {step}: Generator loss: {gen_mean}, Discriminator's loss: {disc_mean}")
				show_tensor_images(fake)
				show_tensor_images(real)
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
			# torch.Tensor(disc_losses[:num_examples]).view(-1, step_bins).mean(1),
			range(len(disc_losses)),
			disc_losses,
			label="Discriminator Loss"
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


# FIT WITHOUT CHECKPOINTS ####
# fit()

# USE THIS FIT VERSION FOR FITTING WITH CHECKPOINTS ###
fit(checkpt=ENABLE_CHECKPOINT, loadcheckptpath=CHECKPOINTFILEPATH)


# ================= #
#  SAVE MODEL...    #
# ================= #
def save_model():
	torch.save(disc.state_dict(), DISC_PATH)
	torch.save(gen.state_dict(), GEN_PATH)


save_model()

