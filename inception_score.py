import numpy as np
import torch
import torchvision


def getIS(p_yx, eps=(1E-16)):
	# get inception score for p(y|x)
	# p_yx => p(y|x), where p_yx[y][x] = p(y|x)
	# eps is  arbitrarily small non-zero value
	# p(y) by averaging over x
	p_y = p_yx.mean(axis=0)
	p_y = np.expand_dims(p_y,0)                           # make dimensions of p_yx and p_y compatible
	# KL-divergence for each x
	KL = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))  # use eps to deal with possible zeros
	# sum KL divergences over classes, and average over images
	KLsum = KL.sum(axis=1)
	KLavg = np.mean(KLsum)
	is_score = np.exp(KLavg)                              # undo the logs
	return is_score


def inception_score(data, cuda=False, batch_size=32, resize=False, n_sec=1, eps=1E-16, print_tag=False):
	# data is the set of images
	N = len(data)
	# Set up CUDA option
	if cuda:
		dtype = torch.cuda.FloatTensor
	else:
		if torch.cuda.is_available():
			print("You should use cuda=True, you can do it!")
		dtype = torch.FloatTensor
	# set up data loader
	loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
	# set up Inception V3 model
	model = torchvision.models.inception_v3(pretrained=True, transform_input=False).type(dtype)
	model.eval()
	up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)

	# helpful prediction function
	def predict(x):
		if resize:
			x = up(x)
		x = model(x)
		return torch.nn.functional.softmax(x, dim=1).data.cpu().numpy()

	# Do predictions
	all_pyx = np.zeros((N,1000))
	for i, batch in enumerate(loader, 0):
		if print_tag:
			print("Batch " + str(i))
		batch = batch.type(dtype)
		batchv = torch.autograd.Variable(batch)
		batch_size_i = batch.size()[0]
		if print_tag:
			print("Built batch")
		# actual prediction:                  Use batch_size_i to handle uneven batch
		all_pyx[i*batch_size:i * batch_size + batch_size_i] = predict(batchv)
		if print_tag:
			print("Predicted batch")

	# Split and compute scores
	sec_scores = []

	for i in range(n_sec):
		if print_tag:
			print("Section " + str(i))
		sec = all_pyx[i * (N // n_sec):(i+1) * (N // n_sec), :]
		sec_scores.append(getIS(sec, eps))

	return np.mean(sec_scores), np.std(sec_scores)

