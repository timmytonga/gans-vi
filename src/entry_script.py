import argparse
import torchvision.transforms as transforms
import torchvision
import torch
import os
import datetime
import utils
import optimizers.optim.extragradient as ExtraGradient
import optimizers.optim.omd as OMD
import optimizers.adasls.adasls as adasls
import optimizers.adasls.sls as sls

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
                                                lr=opt_dict["DLR"],
                                                betas=(opt_dict["BETA1"], opt_dict["BETA2"]))
        gen_optimizer = ExtraGradient.ExtraAdam(generator.parameters(),
                                                lr=opt_dict["GLR"],
                                                betas=(opt_dict["BETA1"], opt_dict["BETA2"]))
    elif opt_dict["name"] == "optimisticadam":
        dis_optimizer = OMD.OptimisticAdam(discriminator.parameters(),
                                           lr=opt_dict["DLR"],
                                           betas=(opt_dict["BETA1"], opt_dict["BETA2"]))
        gen_optimizer = OMD.OptimisticAdam(generator.parameters(),
                                           lr=opt_dict["GLR"],
                                           betas=(opt_dict["BETA1"], opt_dict["BETA2"]))
    elif opt_name == "adaptive_first":

        gen_optimizer = adasls.AdaSLS(gen.parameters(),
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
        dis_optimizer = adasls.AdaSLS(dis.parameters(),
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

        gen_optimizer = sls.Sls(gen.parameters(),
                      c=c,
                      n_batches_per_epoch=n_batches_per_epoch,
                      init_step_size=opt_dict.get("init_step_size", 1),
                      line_search_fn=opt_dict.get("line_search_fn", "armijo"),
                      gamma=opt_dict.get("gamma", 2.0),
                      reset_option=opt_dict.get("reset_option", 1),
                      eta_max=opt_dict.get("eta_max"))

        dis_optimizer = sls.Sls(dis.parameters(),
                      c=c,
                      n_batches_per_epoch=n_batches_per_epoch,
                      init_step_size=opt_dict.get("init_step_size", 1),
                      line_search_fn=opt_dict.get("line_search_fn", "armijo"),
                      gamma=opt_dict.get("gamma", 2.0),
                      reset_option=opt_dict.get("reset_option", 1),
                      eta_max=opt_dict.get("eta_max"))

    elif opt_name == "sgd_goldstein":
        gen_optimizer = sls.Sls(gen.parameters(),
                              c=opt_dict.get("c") or 0.1,
                              reset_option=opt_dict.get("reset_option") or 0,
                              n_batches_per_epoch=n_batches_per_epoch,
                              line_search_fn="goldstein")

        dis_optimizer = sls.Sls(dis.parameters(),
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


def extra_adam_runner(trainloader, generator, discriminator, optim_params, output_path, args):

    dis_optimizer, gen_optimizer = retrieve_optimizer(optim_params, generator, discriminator)
    print("Training Optimizer ({}) on ({})".format(optim_params["name"], args.model))

    gen_updates = 0
    current_iter = 0
    while gen_updates < args.numiter:
        penalty = torch.autograd.Variable([0.])
        if args.cuda:
            penalty = penalty.cuda(0)
        for i, data in enumerate(trainloader):
            x_true, _ = data
            x_true = torch.autograd.Variable(x_true)
            if args.cuda:
                x_true = x_true.cuda(0)

            z = torch.autograd.Variable(utils.sample("normal", (len(x_true), args.nz)))
            if args.cuda:
                z = z.cuda(0)

            x_gen = gen(z)
            p_true, p_gen = dis(x_true), dis(x_gen)
            gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=args.mode)
            dis_loss = - gen_loss.clone()

            if args.gp != 0:
                dis_loss += args.gp * dis.get_penalty(x_true.data, x_gen.data)

            # Discriminator Update
            for p in gen.parameters():
                p.requires_grad = False

            dis_optimizer.zero_grad()
            dis_loss.backward(retain_graph=True)

            step(optim_params["name"], dis_optimizer, current_iter)

            # Genereator Update
            for p in gen.parameters():
                p.requires_grad = True

            for p in dis.parameters():
                p.requires_grad = False

            gen_optimizer.zero_grad()
            gen_loss.backward()

            gen_updates += step(optim_params["name"], dis_optimizer, current_iter)

            for p in dis.parameters():
                p.requires_grad = True

            if args.mode == "wgan" and args.gp == 0:
                for p in dis.parameters():
                    p.data.clamp_(-args.clip, args.clip)




ag = argparse.ArgumentParser()

print("CURRENT WORKING DIRECTIONR: {}".format(os.getcwd()))

# Run parameters
ag.add_argument("--cuda", action="store_true")
ag.add_argument("--datadir", default="datadir")
ag.add_argument("--dataset", choices=("cifar10",), default="cifar10")

# Model parameters
ag.add_argument("--model", choices=('dcgan',), default='dcgan')
ag.add_argument("--mode", choices=('wgan',), default='wgan')
ag.add_argument("--batchsize", default=64, type=int)
ag.add_argument("--numiter", default=500000, type=int)
ag.add_argument("--nfg", default=128, type=int)
ag.add_argument("--nfd", default=128, type=int)
ag.add_argument("--nz", default=128, type=int)
ag.add_argument("--nc", default=3, type=int)
ag.add_argument("--batch_norm_g", default=True, type=bool)
ag.add_argument("--batch_norm_d", default=True, type=bool)
ag.add_argument("--gp", default=10, type=float)
ag.add_argument("--clip", default=0.01, type=float)

# Optimizer parameters
ag.add_argument("--optimizerJSON", choices=("extraadam"), default="extraadam")

dargs = {
    # Run parameters
    "cuda": "true",
    "datadir": "datadir",
    "dataset": "cifar10",
    "seed": 140,
    "outdir": "outdir",

    # model parameters
    "model": 'dcgan',
    "mode": 'wgan',
    "batchsize": 64,
    "numiter": 500000,
    "nfg": 128,
    "nfd": 128,
    "nz": 128,
    "nc": 3,
    "batch_norm_g": True,
    "batch_norm_d": True,
    "gp": 10.0,
    "clip": 0.01,

    # optimizer parameters
    "optimizer": "extraadam"
}

args = argparse.Namespace(**dargs)

dir_name = os.path.join("o{}".format(args.optimizer),
                        "m{}".format(args.model),
                        "t{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
output_dir = os.path.join(args.outdir, dir_name)


def setup_dirs():
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    os.mkdir(output_dir)

setup_dirs()

def get_dataset(name: str, train: bool):
    if name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        dset = torchvision.datasets.CIFAR10(root=args.datadir,
                                            train=train,
                                            transform=transform,
                                            download=True)
        dloader = torch.utils.data.DataLoader(dset,
                                              batch_size=args.batchsize,
                                              shuffle=True,
                                              num_workers=1)

        return dloader


training_set = get_dataset(args.dataset, True)
test_set = get_dataset(args.dataset, False)


if args.model == "resnet":
    gen = ResNet32Generator(args.nz, args.nc, args.nfg, args.batch_norm_g)
    dis = ResNet32Discriminator(args.nc, 1, args.nfd, args.batch_norm_d)
elif args.model == "dcgan":
    gen = DCGAN32Generator(args.nz, args.nc, args.nfg, batchnorm=args.batch_norm_g)
    dis = DCGAN32Discriminator(args.nc, 1, args.nfd, batchnorm=args.batch_norm_d)

if args.cuda:
    gen = gen.cuda(0)
    dis = dis.cuda(0)

gen.apply(lambda x: utils.weight_init(x, mode='normal'))
dis.apply(lambda x: utils.weight_init(x, mode='normal'))