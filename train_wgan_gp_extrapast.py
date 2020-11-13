import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
# torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from dcgan_models import Generator, Discriminator
from wgan_gp_loss import disc_loss_wgan_gp, gen_loss_wgan_gp
from datetime import datetime     # to get time stamp for checkpts
import pytz
from utils import make_noise, show_tensor_images, weights_init, get_cur_timestamp, weights_init_zero
from fit_extrapastsgd import fit_extrapastsgd


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
gen2 = Generator().to(device)
disc2 = Discriminator(wgan_gp=wgan_gp).to(device) 

# UNCOMMENT/COMMENT BELOW FOR LOADING MODEL
# gen, disc = load_model(device)

# =========================================================================== #
#                   HYPERPARAMETERS AND TRAINING DATA                         #
# =========================================================================== #
z_dim = 128
lr = 0.002      # learning rate
if wgan_gp:       # according to varineq paper
	lr = 0.001
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


trainset = DataLoader(CIFAR10(root='.', download=True, transform=transform_train), 
                      batch_size=batch_size, shuffle=True)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
gen2 = gen2.apply(weights_init_zero)
disc2 = disc2.apply(weights_init_zero)


# =========================================================================== #
#                  SET OPTIMIZERS HERE                                          #
# =========================================================================== #
# define the Adam optimizers
gen_opt = torch.optim.SGD(gen.parameters(), lr=lr)
disc_opt = torch.optim.SGD(disc.parameters(), lr=lr)


# =========================================================================== #
#                  SET LOSS FUNCTIONS FOR FITTING                             #
# =========================================================================== #
disc_loss_fn = disc_loss_wgan_gp
gen_loss_fn = gen_loss_wgan_gp





# =========================================================================== #
#                  FITTING                                                    #
# =========================================================================== #
display_step = 1  # for displaying info after this many step (i.e. batch)
disc_repeats = 1  # how many times to train the discriminator per one generator
check_point_after_epochs = 10  # checkpointing every these epochs
n_epochs = 1
save_dict = None  # backup checkpoint for manual saving
torch.manual_seed(0)
validation_noise = make_noise(25, z_dim, device=device)  # so we have a fixed validation noise
torch.manual_seed(int(time.time()))  # this should be random enough for fitting


# FIT WITHOUT CHECKPOINTS ####
fit_extrapastsgd(gen, gen2, gen_opt, disc, disc2, disc_opt, device, n_epochs, trainset, disc_repeats, make_noise, z_dim, disc_loss_fn, gen_loss_fn, display_step, c_lambda, lr, validation_noise)

# USE THIS FIT VERSION FOR FITTING WITH CHECKPOINTS ###
#fit(checkpt=ENABLE_CHECKPOINT, loadcheckptpath=CHECKPOINTFILEPATH)

# ================= #
#  SAVE MODEL...    #
# ================= #
def save_model():
	torch.save(disc.state_dict(), DISC_PATH)
	torch.save(gen.state_dict(), GEN_PATH)


save_model()

