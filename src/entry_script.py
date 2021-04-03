import argparse
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
import optimizers.adasls.sls as sls
import tqdm

from torch.autograd import Variable

from models.dcgan import DCGAN32Generator, DCGAN32Discriminator
from models.resnet import ResNet32Generator, ResNet32Discriminator


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
    elif opt_dict["name"] == "optimisticadam":
        dis_optimizer = OMD.OptimisticAdam(discriminator.parameters(),
                                           lr=opt_dict["learning_rate_dis"],
                                           betas=(opt_dict["beta1"], opt_dict["beta2"]))
        gen_optimizer = OMD.OptimisticAdam(generator.parameters(),
                                           lr=opt_dict["learning_rate_gen"],
                                           betas=(opt_dict["beta1"], opt_dict["beta2"]))
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

    elif opt_name == "sgd_armijo":
        # if opt_dict.get("infer_c"):
        #     c = (1e-3) * np.sqrt(n_batches_per_epoch)
        if opt_dict['c'] == 'theory':
            c = (n_train - batch_size) / (2 * batch_size * (n_train - 1))
        else:
            c = opt_dict.get("c") or 0.1

        gen_optimizer = sls.Sls(generator.parameters(),
                      c=c,
                      n_batches_per_epoch=n_batches_per_epoch,
                      init_step_size=opt_dict.get("init_step_size", 1),
                      line_search_fn=opt_dict.get("line_search_fn", "armijo"),
                      gamma=opt_dict.get("gamma", 2.0),
                      reset_option=opt_dict.get("reset_option", 1),
                      eta_max=opt_dict.get("eta_max"))

        dis_optimizer = sls.Sls(discriminator.parameters(),
                      c=c,
                      n_batches_per_epoch=n_batches_per_epoch,
                      init_step_size=opt_dict.get("init_step_size", 1),
                      line_search_fn=opt_dict.get("line_search_fn", "armijo"),
                      gamma=opt_dict.get("gamma", 2.0),
                      reset_option=opt_dict.get("reset_option", 1),
                      eta_max=opt_dict.get("eta_max"))

    elif opt_name == "sgd_goldstein":
        gen_optimizer = sls.Sls(generator.parameters(),
                              c=opt_dict.get("c") or 0.1,
                              reset_option=opt_dict.get("reset_option") or 0,
                              n_batches_per_epoch=n_batches_per_epoch,
                              line_search_fn="goldstein")

        dis_optimizer = sls.Sls(discriminator.parameters(),
                              c=opt_dict.get("c") or 0.1,
                              reset_option=opt_dict.get("reset_option") or 0,
                              n_batches_per_epoch=n_batches_per_epoch,
                              line_search_fn="goldstein")
    else:
        raise AssertionError("Failed to retrieve optimizer: No optimizer of name: {}", opt_dict["name"])

    return dis_optimizer, gen_optimizer


def step(opt_name, optimizer, ts):
    if opt_name == "extraadam":
        if (ts + 1) % 2 != 0:
            optimizer.extrapolation()
            return 0
        else:
            optimizer.step()
            return 1
    elif opt_name == "pastadam" or opt_name == "optimisticadam":
        optimizer.step()
        return 1


def runner(trainloader, generator, discriminator, optim_params, output_path, model_params, cuda):

    dis_optimizer, gen_optimizer = retrieve_optimizer(optim_params, generator, discriminator, len(trainloader), model_params["batch_size"])
    print("Training Optimizer ({}) on ({})".format(optim_params["name"], model_params["model"]))

    N_SAMPLES = 50000
    RESOLUTION = 32
    EVAL_FREQ = 10000

    gen_updates = 0
    current_iter = 0
    epoch = 1
    while gen_updates < model_params["num_iter"]:
        penalty = Variable(torch.Tensor([0.]))
        if cuda:
            penalty = penalty.cuda(0)
        loop = tqdm.tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        for i, data in loop:
            x_true, _ = data
            x_true = torch.autograd.Variable(x_true)
            if cuda:
                x_true = x_true.cuda(0)

            z = torch.autograd.Variable(utils.sample("normal", (len(x_true), model_params["num_latent"])))
            if cuda:
                z = z.cuda(0)

            x_gen = generator(z)
            p_true, p_gen = discriminator(x_true), discriminator(x_gen)
            gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=model_params["mode"])
            dis_loss = - gen_loss.clone()

            if model_params["gradient_penalty"] != 0:
                penalty = discriminator.get_penalty(x_true.data, x_gen.data)
                dis_loss += model_params["gradient_penalty"] * penalty

            # Discriminator Update
            for p in generator.parameters():
                p.requires_grad = False

            dis_optimizer.zero_grad()
            dis_loss.backward(retain_graph=True)

            step(optim_params["name"], dis_optimizer, current_iter)

            # Generaator Update
            for p in generator.parameters():
                p.requires_grad = True

            for p in discriminator.parameters():
                p.requires_grad = False

            gen_optimizer.zero_grad()
            gen_loss.backward()

            gen_updates += step(optim_params["name"], gen_optimizer, current_iter)

            for p in discriminator.parameters():
                p.requires_grad = True

            if model_params["mode"] == "wgan" and model_params["gradient_penalty"] == 0:
                for p in discriminator.parameters():
                    p.data.clamp_(-model_params["clip"], model_params["clip"])

            current_iter += 1

            loop.set_description(f"EPOCH: {epoch}")
            loop.set_postfix(GEN_UPDATE="{}/{}".format(gen_updates, model_params["num_iter"]),
                             DISLOSS=dis_loss.item(),
                             GENLOSS=gen_loss.item())
        epoch += 1


def run_config(all_params, dataset: str):
    model_params = all_params["model_params"]
    opt_params = all_params["optimizer_params"]

    def get_or_error(dic, key_name):
        temp_value = dic.get(key_name)
        if temp_value is not None:
            return temp_value

        raise KeyError(f"Missing required parameter {key_name} in model_params for given config file.")


    OPTIMIZER_NAME = opt_params["name"]
    MODEL = get_or_error(model_params, "model")
    N_LATENT = get_or_error(model_params, "num_latent")
    N_FILTERS_G = get_or_error(model_params, "num_filters_gen")
    N_FILTERS_D = get_or_error(model_params, "num_filters_dis")
    DISTRIBUTION = get_or_error(model_params, "distribution")
    BATCH_NORM_G = True
    BATCH_NORM_D = get_or_error(model_params, "batchnorm_dis")
    N_CHANNEL = 3
    CUDA = True
    OUTDIR = "outdir"
    DATADIR = "datadir"
    SEED = get_or_error(model_params, "seed")
    torch.manual_seed(SEED)

    dir_name = os.path.join("o{}".format(OPTIMIZER_NAME),
                            "m{}".format(MODEL),
                            "t{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    output_dir = os.path.join(OUTDIR, dir_name)


    def setup_dirs():
        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)
        print(f"Making output directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    setup_dirs()


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
        gen = ResNet32Generator(n_in=N_LATENT, n_out=N_CHANNEL, num_filters=N_FILTERS_G, batchnorm=BATCH_NORM_G)
        dis = ResNet32Discriminator(n_in=N_CHANNEL, n_out=1, num_filters=N_FILTERS_D, batchnorm=BATCH_NORM_D)
    elif MODEL == "dcgan":
        gen = DCGAN32Generator(n_in=N_LATENT, n_out=N_CHANNEL, n_filters=N_FILTERS_G, batchnorm=BATCH_NORM_G)
        dis = DCGAN32Discriminator(n_in=N_CHANNEL, n_out=1, n_filters=N_FILTERS_D, batchnorm=BATCH_NORM_D)
    else:
        raise KeyError(f"{MODEL} model not recognized")

    if CUDA:
        gen = gen.cuda(0)
        dis = dis.cuda(0)

    gen.apply(lambda x: utils.weight_init(x, mode=DISTRIBUTION))
    dis.apply(lambda x: utils.weight_init(x, mode=DISTRIBUTION))

    runner(training_set, gen, dis, opt_params, output_dir, model_params, CUDA)


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("CUDA is not enabled; enable CUDA for pytorch in order to run script")
        exit()

    print("CURRENT WORKING DIRECTORY: {}".format(os.getcwd()))

    with open("../config/default_dcgan_wgan_extraadam.json") as f:
        all_params = json.load(f)

    run_config(all_params, "cifar10")