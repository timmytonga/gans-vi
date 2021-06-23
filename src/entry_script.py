import torchvision.transforms as transforms
import torchvision
import torch
import os
import datetime
import utils
import json
import optimizers.optim.extragradient as ExtraGradient
import optimizers.optim.omd as OMD
import optimizers.adasls.adasls as adasls
import optimizers.optim.adapeg as AdaPEG
import optimizers.optim.torch_svg as SVRG
import numpy as np
import tqdm
import wandb
import time

from torch.autograd import Variable
from torch.nn import functional as F

from models.dcgan import DCGAN32Generator, DCGAN32Discriminator
from models.resnet import ResNet32Generator, ResNet32Discriminator
from scipy.stats import entropy
from src.pytorch_fid.fid import calculate_fid_given_paths
from src.optimizers.optim.run_vr import recalibrate


def inception_score(imgs, cuda=True, batch_size=100, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = torchvision.models.inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()

    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        with torch.no_grad():
            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def retrieve_optimizer(opt_dict,
                       generator,
                       discriminator,
                       n_train,
                       batch_size):
    opt_name = opt_dict["name"]
    n_batches_per_epoch = n_train / batch_size
    if opt_dict["name"] == "extraadam" or opt_dict["name"] == "pastadam":
        dis_optimizer = ExtraGradient.ExtraAdam(discriminator.parameters(),
                                                lr=opt_dict["learning_rate_dis"],
                                                betas=(opt_dict["beta1"], opt_dict["beta2"]))
        gen_optimizer = ExtraGradient.ExtraAdam(generator.parameters(),
                                                lr=opt_dict["learning_rate_gen"],
                                                betas=(opt_dict["beta1"], opt_dict["beta2"]))
    elif opt_dict["name"] == "adapeg":
        dis_optimizer = AdaPEG.AdaPEGAdam(discriminator.parameters(),
                                          lr=opt_dict["learning_rate_dis"],
                                          betas=(opt_dict["beta1"], opt_dict["beta2"]),
                                          squared_grad=opt_dict["squared_grad"],
                                          optimistic=opt_dict["optimistic"])
        gen_optimizer = AdaPEG.AdaPEGAdam(generator.parameters(),
                                          lr=opt_dict["learning_rate_gen"],
                                          betas=(opt_dict["beta1"], opt_dict["beta2"]),
                                          squared_grad=opt_dict["squared_grad"],
                                          optimistic=opt_dict["optimistic"])
    elif opt_dict["name"] == "optimisticadam":
        dis_optimizer = OMD.OptimisticAdam(discriminator.parameters(),
                                           lr=opt_dict["learning_rate_dis"],
                                           betas=(opt_dict["beta1"], opt_dict["beta2"]))
        gen_optimizer = OMD.OptimisticAdam(generator.parameters(),
                                           lr=opt_dict["learning_rate_gen"],
                                           betas=(opt_dict["beta1"], opt_dict["beta2"]))
    elif opt_dict["name"] == "adam":
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=opt_dict["learning_rate_dis"],
                                         betas=(opt_dict["beta1"], opt_dict["beta2"]))
        gen_optimizer = torch.optim.Adam(generator.parameters(),
                                         lr=opt_dict["learning_rate_gen"],
                                         betas=(opt_dict["beta1"], opt_dict["beta2"]))
    elif opt_dict["name"] == "svrg":
        dis_optimizer = SVRG.SVRG(discriminator.parameters(),
                                  vr_from_epoch=opt_dict["vr_after"],
                                  nbatches=n_train,
                                  lr=opt_dict["learning_rate_dis"],
                                  momentum=opt_dict["momentum"],
                                  weight_decay=opt_dict["weight_decay"])

        gen_optimizer = SVRG.SVRG(generator.parameters(),
                                  vr_from_epoch=opt_dict["vr_after"],
                                  nbatches=n_train,
                                  lr=opt_dict["learning_rate_gen"],
                                  momentum=opt_dict["momentum"],
                                  weight_decay=opt_dict["weight_decay"])

    elif opt_name == "adaptive_first":

        gen_optimizer = adasls.AdaSLS(generator.parameters(),
                     c=opt_dict['c'],
                     n_batches_per_epoch=n_batches_per_epoch,
                     gv_option=opt_dict.get('gv_option', 'per_param'),
                     base_opt=opt_dict['base_opt'],
                     pp_norm_method=opt_dict['pp_norm_method'],
                     momentum=opt_dict.get('momentum', 0),
                     beta=opt_dict.get('beta', 0.99),
                     gamma=opt_dict.get('gamma', 2),
                     init_step_size=opt_dict.get('init_step_size', 1),
                     adapt_flag=opt_dict.get('adapt_flag', 'constant'),
                     step_size_method=opt_dict['step_size_method'],
                     # sls stuff
                     beta_b=opt_dict.get('beta_b', .9),
                     beta_f=opt_dict.get('beta_f', 2.),
                     reset_option=opt_dict.get('reset_option', 1),
                     line_search_fn=opt_dict.get('line_search_fn', "armijo"),
                     mom_type=opt_dict.get('mom_type', "standard"),
                     )
        dis_optimizer = adasls.AdaSLS(discriminator.parameters(),
                     c=opt_dict['c'],
                     n_batches_per_epoch=n_batches_per_epoch,
                     gv_option=opt_dict.get('gv_option', 'per_param'),
                     base_opt=opt_dict['base_opt'],
                     pp_norm_method=opt_dict['pp_norm_method'],
                     momentum=opt_dict.get('momentum', 0),
                     beta=opt_dict.get('beta', 0.99),
                     gamma=opt_dict.get('gamma', 2),
                     init_step_size=opt_dict.get('init_step_size', 1),
                     adapt_flag=opt_dict.get('adapt_flag', 'constant'),
                     step_size_method=opt_dict['step_size_method'],
                     # sls stuff
                     beta_b=opt_dict.get('beta_b', .9),
                     beta_f=opt_dict.get('beta_f', 2.),
                     reset_option=opt_dict.get('reset_option', 1),
                     line_search_fn=opt_dict.get('line_search_fn', "armijo"),
                     mom_type=opt_dict.get('mom_type', "standard"),
                     )
    else:
        raise AssertionError("Failed to retrieve optimizer: No optimizer of name {}", opt_dict["name"])

    return dis_optimizer, gen_optimizer


def step(opt_params, optimizer, ts, loss, epoch, lr, batch_id, retain_graph=False):
    if opt_params["name"] == "extraadam":
        loss.backward(retain_graph=retain_graph)
        if (ts + 1) % 2 != 0:
            optimizer.extrapolation()
            return 0
        else:
            optimizer.step()
            return 1
    elif opt_params["name"] == "pastadam" or opt_params["name"] == "optimisticadam" or opt_params["name"] == "adam" or opt_params["name"] == "adapeg":
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        return 1
    elif opt_params["name"] == "adaptive_first":
        closure = lambda: loss
        optimizer.step(closure=closure, retain_graph=retain_graph)
        return 1
    elif opt_params["name"] == "svrg":
        if opt_params["lr_reduction"] == "default":
            lr = lr * (0.1 ** (epoch // 75))
        elif opt_params["lr_reduction"] == "none" or opt_params["lr_reduction"] == "False":
            lr = lr
        elif opt_params["lr_reduction"] == "150":
            lr = lr * (0.1 ** (epoch // 150))
        elif opt_params["lr_reduction"] == "150-225":
            lr = lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
        elif opt_params["lr_reduction"] == "up5x-20-down150":
            if epoch < 20:
                lr = lr
            else:
                lr = 3.0 * lr * (0.1 ** (epoch // 150))
        elif opt_params["lr_reduction"] == "up30-150-225":
            if epoch < 30:
                lr = lr
            else:
                lr = 3.0 * lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
        elif opt_params["lr_reduction"] == "every30":
            lr = lr * (0.1 ** (epoch // 30))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        closure = lambda: loss
        loss.backward(retain_graph=retain_graph)
        optimizer.step(batch_id, closure)

    else:
        raise RuntimeError("Could not find step procedure for optimizer {}".format(opt_params["name"]))


def runner(trainloader, generator, discriminator, optim_params, model_params, device):

    dis_optimizer, gen_optimizer = retrieve_optimizer(optim_params, generator, discriminator, len(trainloader), model_params["batch_size"])
    print("Training Optimizer ({}) on ({})".format(optim_params["name"], model_params["model"]))

    gen_updates = 0
    o_gen_updates = None
    current_iter = 0
    epoch = 1
    postfix_kwargs = {}

    gen_param_avg = []
    param_temp_holder = []
    for i, param in enumerate(generator.parameters()):
        gen_param_avg.append(param.data.clone())
        param_temp_holder.append(None)

    begin_time = time.time()
    while gen_updates < model_params["num_iter"]:
        penalty = Variable(torch.Tensor([0.]))
        penalty = penalty.to(device=device)
        loop = tqdm.tqdm(enumerate(trainloader), total=len(trainloader), leave=False)

        loop.set_description(f"EPOCH: {epoch}")
        if optim_params["name"] == "svrg":
            if epoch >= 1:
                recalibrate(model_params=model_params,
                            train_loader=trainloader,
                            generator=generator,
                            discriminator=discriminator,
                            gen_optimizer=gen_optimizer,
                            dis_optimizer=dis_optimizer,
                            device=device)

        for i, data in loop:
            x_true, _ = data
            x_true = torch.autograd.Variable(x_true)

            z = torch.autograd.Variable(utils.sample(model_params["distribution"], (len(x_true), model_params["num_latent"])))
            x_true = x_true.to(device)
            z = z.to(device)

            if optim_params["name"] == "pastadam":
                dis_optimizer.extrapolation()
                gen_optimizer.extrapolation()

                if model_params["mode"] == "wgan" and model_params["gradient_penalty"] == 0.0:
                    for p in discriminator.parameters():
                        p.data.clamp_(-model_params["CLIP"], model_params["CLIP"])

            x_gen = generator(z)


            if optim_params["name"] != "adam" and optim_params["name"] != "adaptive_first" and optim_params["name"] != "svrg":
                p_true, p_gen = discriminator(x_true), discriminator(x_gen)
                gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=model_params["mode"])
                dis_loss = - gen_loss.clone()

                postfix_kwargs = {"GEN_UPDATE":"{}/{}".format(gen_updates, model_params["num_iter"]),
                                 "DISLOSS": dis_loss.item(),
                                 "GENLOSS":gen_loss.item()}

                if model_params["gradient_penalty"] != 0:
                    penalty = discriminator.get_penalty(x_true.data, x_gen.data)
                    dis_loss += model_params["gradient_penalty"] * penalty

                # Discriminator Update
                for p in generator.parameters():
                    p.requires_grad = False

                dis_optimizer.zero_grad()

                step(optim_params, dis_optimizer, current_iter, dis_loss, epoch, optim_params["learning_rate_dis"], i, retain_graph=True)

                # Generaator Update
                for p in generator.parameters():
                    p.requires_grad = True

                for p in discriminator.parameters():
                    p.requires_grad = False

                gen_optimizer.zero_grad()

                gen_updates += step(optim_params, gen_optimizer, current_iter, gen_loss, epoch, optim_params["learning_rate_gen"], i)

                for param_i, param in enumerate(generator.parameters()):
                    gen_param_avg[param_i] = gen_param_avg[param_i] * gen_updates / (gen_updates + 1.) + param.data.clone() / (gen_updates + 1.)

                for p in discriminator.parameters():
                    p.requires_grad = True

                if model_params["mode"] == "wgan" and model_params["gradient_penalty"] == 0:
                    for p in discriminator.parameters():
                        p.data.clamp_(-model_params["clip"], model_params["clip"])

                current_iter += 1

            else:


                if model_params["update_frequency"] == 1 or (current_iter+1)%model_params["update_frequency"] != 0:
                    for p in generator.parameters():
                        p.requires_grad = False

                    p_true, p_gen = discriminator(x_true), discriminator(x_gen)
                    dis_loss = - utils.compute_gan_loss(p_true, p_gen, mode=model_params["mode"])
                    postfix_kwargs["DISLOSS"] = dis_loss.item()
                    if model_params["gradient_penalty"] != 0:
                        penalty = discriminator.get_penalty(x_true.data, x_gen.data)
                        dis_loss += penalty * model_params["gradient_penalty"]

                    dis_optimizer.zero_grad()

                    step(optim_params, dis_optimizer, current_iter, dis_loss, epoch, optim_params["learning_rate_dis"], i, retain_graph=True if model_params["update_frequency"] == 1 else False)

                    if model_params["mode"] == "wgan" and model_params["gradient_penalty"] == 0.0:
                        for p in discriminator.parameters():
                            p.data.clamp_(-model_params["clip"], model_params["clip"])

                    for p in generator.parameters():
                        p.requires_grad = True

                if model_params["update_frequency"] == 1 or (current_iter+1)%model_params["update_frequency"] == 0:
                    for p in discriminator.parameters():
                        p.requires_grad = False

                    p_true, p_gen = discriminator(x_true), discriminator(x_gen)
                    gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=model_params["mode"])
                    postfix_kwargs["GENLOSS"] = gen_loss.item()
                    gen_optimizer.zero_grad()

                    step(optim_params, gen_optimizer, current_iter, gen_loss, epoch, optim_params["learning_rate_gen"], i)

                    for param_i, param in enumerate(generator.parameters()):
                        gen_param_avg[param_i] = gen_param_avg[param_i] * gen_updates / (gen_updates + 1.) + param.data.clone()/(gen_updates + 1.)

                    for p in discriminator.parameters():
                        p.requires_grad = True

                    gen_updates += 1

                postfix_kwargs["GEN_UPDATE"] = "{}/{}".format(gen_updates, model_params["num_iter"])

                current_iter += 1

            loop.set_postfix(**postfix_kwargs)

            if gen_updates % model_params["evaluate_frequency"] == 0 and o_gen_updates != gen_updates:
                o_gen_updates = gen_updates
                torch.cuda.empty_cache()
                if optim_params["average"]:
                    for j, param in enumerate(generator.parameters()):
                        param_temp_holder[j] = param.data
                        param.data = gen_param_avg[j]

                all_samples = []
                samples = torch.randn(model_params["num_samples"], model_params["num_latent"])
                for i in range(0, model_params["num_samples"], 100):
                    samples_100 = samples[i:i+100].to(device=device)
                    all_samples.append(generator(samples_100).cpu().data.numpy())

                all_samples = np.concatenate(all_samples, axis=0)
                # all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')

                inc_is = calculate_fid_given_paths(all_samples, batch_size=100, device=device, dims=2048)

                ex_arr = generator(utils.sample(model_params["distribution"], (100, model_params["num_latent"])).to(device=device))
                ex_images = utils.unormalize(ex_arr)
                wlogdic = {"INCEPTION_SCORE": inc_is,
                           "PASSED_TIME": time.time() - begin_time,
                           "examples": [wandb.Image(utils.image_data(ex_images.data, 10), caption=f"GEN_UPDATE {gen_updates} examples")]}
                wandb.log(wlogdic)

                if optim_params["average"]:
                    for j, param in enumerate(generator.parameters()):
                        param.data = param_temp_holder[j]


        epoch += 1

    optim_name = optim_params["name"]
    torch.save({
        "model_params": generator.state_dict(),
        "optimizer_params": gen_optimizer.state_dict(),
        "optim_config": optim_params
    }, os.path.join("outdir", f"gen_params_{optim_name}_{int(begin_time)}.ckpt"))
    # wandb.save(os.path.join("outdir", "gen_params.ckpt"))



def run_config(all_params, dataset: str, experiment_name: str):
    model_params = all_params["model_params"]
    opt_params = all_params["optimizer_params"]

    def get_or_error(dic, key_name):
        temp_value = dic.get(key_name)
        if temp_value is not None:
            return temp_value

        raise KeyError(f"Missing required parameter {key_name} in model_params for given config file.")

    MODEL = get_or_error(model_params, "model")
    N_LATENT = get_or_error(model_params, "num_latent")
    N_FILTERS_G = get_or_error(model_params, "num_filters_gen")
    N_FILTERS_D = get_or_error(model_params, "num_filters_dis")
    DISTRIBUTION = get_or_error(model_params, "distribution")
    BATCH_NORM_G = True
    BATCH_NORM_D = get_or_error(model_params, "batchnorm_dis")
    N_CHANNEL = 3
    CUDA = 0
    if isinstance(CUDA, int):
        device = torch.device(f"cuda:{CUDA}")
    else:
        device = torch.device("cpu")
    OUTDIR = "outdir"
    DATADIR = "datadir"
    SEED = get_or_error(model_params, "seed")
    # torch.manual_seed(SEED)
    # np.random.seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    dir_name = os.path.join("o{}".format(experiment_name),
                            "m{}".format(MODEL),
                            "t{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    output_dir = os.path.join(OUTDIR, dir_name)


    def setup_dirs():
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
        print(f"Making output directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    #setup_dirs()


    def get_dataset(name: str, train: bool):
        dataset_dir = os.path.join(DATADIR, name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        if name == "cifar10":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            dset = torchvision.datasets.CIFAR10(root=dataset_dir,
                                                train=train,
                                                transform=transform,
                                                download=True)
            dloader = torch.utils.data.DataLoader(dset,
                                                  batch_size=model_params["batch_size"],
                                                  shuffle=True,
                                                  num_workers=1)

            return dloader


    training_set = get_dataset(dataset, True)
    test_set = get_dataset(dataset, False)

    if MODEL == "resnet":
        gen = ResNet32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, BATCH_NORM_G)
        dis = ResNet32Discriminator(N_CHANNEL, 1, N_FILTERS_D, BATCH_NORM_D)
    elif MODEL == "dcgan":
        gen = DCGAN32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, batchnorm=BATCH_NORM_G)
        dis = DCGAN32Discriminator(N_CHANNEL, 1, N_FILTERS_D, batchnorm=BATCH_NORM_D)
    else:
        raise KeyError(f"{MODEL} model not recognized")

    gen = gen.to(device=device)
    dis = dis.to(device=device)

    wandb.watch(gen)

    gen.apply(lambda x: utils.weight_init(x, mode=DISTRIBUTION))
    dis.apply(lambda x: utils.weight_init(x, mode=DISTRIBUTION))

    runner(training_set, gen, dis, opt_params, model_params, device)


def retrieve_line_search_paper_parameters():
    # ------------------ #
    # III. Adaptive with first order preconditioners + line-search/SPS
    # 1. Adaptive + SLS
    # 1.1. Lipschitz + Adagrad
    reset_option_list = [0, 1]

    adaptive_first_sls_lipschitz_list = []
    for reset_option in reset_option_list:
        adaptive_first_sls_lipschitz_list += [{'name': 'adaptive_first',
                                               'c': np.random.uniform(low=0.1, high=0.8),
                                               'gv_option': 'per_param',
                                               'base_opt': 'adagrad',
                                               'pp_norm_method': 'pp_lipschitz',
                                               'init_step_size': 100,
                                               # setting init step-size to 100. SLS should be robust to this
                                               "momentum": 0.,
                                               'step_size_method': 'sls',
                                               'reset_option': reset_option}]

    # 1.2. Armijo + Adam / Amsgrad / Amsgrad
    adaptive_first_sls_armijo_list = []
    reset_option_list = [0, 1]
    base_opt_list = ['adam', 'amsgrad', 'adagrad']

    for base_opt in base_opt_list:
        for reset_option in reset_option_list:
            adaptive_first_sls_armijo_list += [{'name': 'adaptive_first',
                                                'c': np.random.uniform(low=0.1, high=0.55),
                                                'gv_option': 'per_param',
                                                'base_opt': base_opt,
                                                'pp_norm_method': 'pp_armijo',
                                                'init_step_size': 100,
                                                # setting init step-size to 100. SLS should be robust to this
                                                "momentum": 0.,
                                                'step_size_method': 'sls',
                                                'reset_option': reset_option}]


    # 2. Adaptive + SPS / Only Armijo
    base_opt_list = ['adam', 'amsgrad', 'adagrad']

    adaptive_first_sps_list = []
    for base_opt in base_opt_list:
        adaptive_first_sps_list += [{'name': 'adaptive_first',
                                     'c': np.random.uniform(low=0.2, high=1),
                                     'gv_option': 'per_param',
                                     'base_opt': base_opt,
                                     'pp_norm_method': 'pp_armijo',
                                     'init_step_size': 1,
                                     "momentum": 0.,
                                     'step_size_method': 'sps',
                                     'adapt_flag': 'smooth_iter'}]

    # standard momentum
    opt_list = []
    for base_opt in base_opt_list:
        opt_list += [{'name': 'adaptive_first',
                      'mom_type': "standard",
                      'c': np.random.uniform(low=0.1, high=0.2),
                      'gv_option': 'per_param',
                      'base_opt': base_opt,
                      'pp_norm_method': 'pp_armijo',
                      'init_step_size': 100,  # setting init step-size to 100. SLS should be robust to this
                      "momentum": np.random.uniform(low=0, high=0.95),
                      'step_size_method': 'sls',
                      'reset_option': 1}]

    for base_opt in base_opt_list:
        opt_list += [{'name': 'adaptive_first',
                      'mom_type': "heavy_ball",
                      'c': np.random.uniform(low=0.15, high=0.25),
                      'gv_option': 'per_param',
                      'base_opt': base_opt,
                      'pp_norm_method': 'pp_armijo',
                      'init_step_size': 100,  # setting init step-size to 100. SLS should be robust to this
                      "momentum": np.random.uniform(low=0, high=0.95),
                      'step_size_method': 'sls',
                      'reset_option': 1}]

    return opt_list + adaptive_first_sls_lipschitz_list + adaptive_first_sls_lipschitz_list + adaptive_first_sps_list


def get_adapeg_params():
    params = {
        "model_params": {
            "batch_size": 64,
            "model": "dcgan",
            "num_iter": 500000,
            "ema": 0.9999,
            "num_latent": 128,
            "batchnorm_dis": False,
            "optimizer": "adam",
            "clip": 0.01,
            "gradient_penalty": 10,
            "mode": "wgan",
            "seed": 1318,
            "distribution": "normal",
            "initialization": "normal",
            "num_filters_gen": 64,
            "num_filters_dis": 64
        },
        "optimizer_params": []
    }

    for lr in [0.0001, 0.00001]:
        optim_param_base = {
            "name": "adapeg",
            "learning_rate_dis":lr,
            "learning_rate_gen":lr,
            "beta2":0.9,
            "beta1":0.5,
            "squared_grad": True,
            "optimistic": False
        }

        params["optimizer_params"].append(optim_param_base)

    for lr in [0.001, 0.00001]:
        optim_param_base = {
            "name": "adapeg",
            "learning_rate_dis": lr,
            "learning_rate_gen": lr,
            "beta2": 0.9,
            "beta1": 0.5,
            "squared_grad": True,
            "optimistic": True
        }

        params["optimizer_params"].append(optim_param_base)

    return params


def get_svrg_hyperparameters():
    params = {
        "model_params": {
            "batch_size": 64,
            "model": "dcgan",
            "num_iter": 500000,
            "ema": 0.9999,
            "num_latent": 128,
            "batchnorm_dis": False,
            "optimizer": "adam",
            "clip": 0.01,
            "gradient_penalty": 10,
            "mode": "wgan",
            "seed": 1318,
            "distribution": "normal",
            "initialization": "normal",
            "num_filters_gen": 64,
            "num_filters_dis": 64,
            "update_frequency": 1
        },
        "optimizer_params": []
    }

    params["optimizer_params"].append({
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'learning_rate_dis': 0.1,
        'learning_rate_gen': 0.1,
        'lr_reduction': "150-225",
        "name": "svrg",
        "vr_after": 1
    })

    return params


if __name__ == "__main__":

    #torch.autograd.set_detect_anomaly(True)
    if not torch.cuda.is_available():
        print("CUDA is not enabled; enable CUDA for pytorch in order to run script")
        exit()

    print("CURRENT WORKING DIRECTORY: {}".format(os.getcwd()))

    # for opt in retrieve_line_search_paper_parameters():
    #     with open("../config/default_dcgan_wgangp_pastextraadam.json") as f:
    #         all_params = json.load(f)
    #
    #     all_params["model_params"]["num_samples"] = 10000
    #     all_params["model_params"]["evaluate_frequency"] = 10
    #     all_params["model_params"]["num_iter"] = 100000
    #
    #     all_params["optimizer_params"] = opt
    #     all_params["model_params"]["update_frequency"] = 5
    #
    #     print(json.dumps(all_params, indent=4))
    #
    #     run = wandb.init(entity="optimproject", project='optimproj', config=all_params, reinit=True)
    #
    #     run_config(all_params, "cifar10", "textexperiment")
    #
    #     run.finish()


    all_params = get_svrg_hyperparameters()
    for opt_i in all_params["optimizer_params"]:
        inner_params = {}
        inner_params["model_params"] = all_params["model_params"]
        inner_params["optimizer_params"] = opt_i


        inner_params["model_params"]["evaluate_frequency"] = 10
        inner_params["model_params"]["num_samples"] = 500
        inner_params["model_params"]["num_iter"] = 100000
        inner_params["optimizer_params"]["average"] = False
        print(json.dumps(inner_params, indent=4))
        with wandb.init(entity="optimproject", project='optimproj', config=inner_params, reinit=True, mode="disabled") as r:
            run_config(inner_params, "cifar10", "testexperiment")

    # include = {"default_dcgan_wgangp_optimisticextraadam.json"}
    # for file_name in os.listdir("../config"):
    #     if file_name not in include:
    #         continue
    #     with open(os.path.join("../config", file_name)) as f:
    #         all_params = json.load(f)
    #     if all_params["model_params"]["model"] != "resnet":
    #         if all_params["model_params"]["gradient_penalty"] != 0.0:
    #             all_params["model_params"]["evaluate_frequency"] = 2500
    #             all_params["model_params"]["num_samples"] = 50000
    #             all_params["model_params"]["num_iter"] = 100000
    #
    #             all_params["optimizer_params"]["learning_rate_dis"] = 0.0009
    #             all_params["optimizer_params"]["learning_rate_gen"] = 0.0009
    #             all_params["optimizer_params"]["average"] = False
    #             if all_params["optimizer_params"]["name"] == "adam":
    #                 all_params["optimizer_params"]["average"] = False
    #
    #             print(json.dumps(all_params, indent=4))
    #             with wandb.init(entity="optimproject", project='optimproj', config=all_params, reinit=True) as r:
    #                 wandb.save(os.path.join(wandb.run.dir, "*.ckpt"))
    #                 run_config(all_params, "cifar10", "testexperiment")
    #
    #             print("\n\n")



