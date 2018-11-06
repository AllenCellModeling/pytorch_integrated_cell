import argparse
import importlib
import numpy as np
import os
import torch
import datetime
import socket
import json

from integrated_cell import model_utils
from integrated_cell.utils import str2bool
from integrated_cell import utils


# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True


def setup(args):
    if (args["save_parent"] is not None) and (args["save_dir"] is not None):
        raise ValueError(
            "--save_dir and --save_parent are both set. Please choose one or the other."
        )

    if (
        (args["train_module"] is not None) and (args["train_module_pt1"] is not None)
    ) or (
        (args["train_module"] is not None) and (args["train_module_pt2"] is not None)
    ):
        raise ValueError(
            "--train_module and --train_model_pt1 or --train_model_pt2 are both set. Please choose a global train module or specify partial models."
        )

    the_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if args["save_parent"] is not None:
        args["save_dir"] = os.path.join(args["save_parent"], the_time)

    args["the_time"] = the_time
    args["hostname"] = socket.gethostname()

    if args["data_save_path"] is None:
        args["data_save_path"] = args["save_dir"] + os.sep + "data.pkl"

    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

    if args["train_module"] is not None:
        args["train_module_pt1"] = args["train_module"]
        args["train_module_pt2"] = args["train_module"]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ID) for ID in args["gpu_ids"]])
    args["gpu_ids"] = list(range(0, len(args["gpu_ids"])))

    torch.manual_seed(args["myseed"])
    torch.cuda.manual_seed(args["myseed"])
    np.random.seed(args["myseed"])

    if args["nepochs_pt2"] == -1:
        args["nepochs_pt2"] = args["nepochs"]

    return args


def setup_kwargs_data_provider(args):
    data_provider_kwargs = args["kwargs_dp"]

    data_provider_kwargs["save_path"] = args["data_save_path"]
    data_provider_kwargs["batch_size"] = args["batch_size"]
    data_provider_kwargs["im_dir"] = args["imdir"]
    data_provider_kwargs["n_dat"] = args["ndat"]
    data_provider_kwargs["channelInds"] = args["channel_inds"]

    return args["dataProvider"], data_provider_kwargs


def setup_kwargs_network(args):

    kwargs_base = {}
    kwargs_base["network_name"] = args["network_name"]
    kwargs_base["optim_name"] = args["optimizer"]
    kwargs_base["gpu_ids"] = args["gpu_ids"]

    # Encoder
    kwargs_enc = kwargs_base.copy()
    kwargs_enc["save_path"] = "{}/{}.pth".format(args["save_dir"], "enc")
    # kwargs_enc['save_args_path'] = '{}/args_{}.json'.format(args['save_dir'], 'enc')
    kwargs_enc["component_name"] = "Enc"
    kwargs_enc["network_name"] = args["network_name"]
    kwargs_enc["kwargs_network"] = args["kwargs_enc"]
    kwargs_enc["kwargs_network"]["n_channels"] = len(args["channels"])
    kwargs_enc["kwargs_network"]["n_classes"] = args["n_classes"]
    kwargs_enc["kwargs_network"]["n_ref"] = args["n_ref"]
    kwargs_enc["kwargs_network"]["n_latent_dim"] = args["n_latent_dim"]

    kwargs_enc["kwargs_optim"] = args["kwargs_enc_optim"]
    kwargs_enc["kwargs_optim"]["lr"] = args["lr_enc"]

    # Decoder
    kwargs_dec = kwargs_base.copy()
    kwargs_dec["save_path"] = "{}/{}.pth".format(args["save_dir"], "dec")
    # kwargs_enc['save_args_path'] = '{}/args_{}.json'.format(args['save_dir'], 'dec')
    kwargs_dec["component_name"] = "Dec"
    kwargs_dec["network_name"] = args["network_name"]
    kwargs_dec["kwargs_network"] = args["kwargs_dec"]
    kwargs_dec["kwargs_network"]["n_channels"] = len(args["channels"])
    kwargs_dec["kwargs_network"]["n_classes"] = args["n_classes"]
    kwargs_dec["kwargs_network"]["n_ref"] = args["n_ref"]
    kwargs_dec["kwargs_network"]["n_latent_dim"] = args["n_latent_dim"]

    kwargs_dec["kwargs_optim"] = args["kwargs_dec_optim"]
    kwargs_dec["kwargs_optim"]["lr"] = args["lr_dec"]

    # Encoder discriminator
    kwargs_encD = kwargs_base.copy()
    kwargs_encD["save_path"] = "{}/{}.pth".format(args["save_dir"], "encD")
    # kwargs_enc['save_args_path'] = '{}/args_{}.json'.format(args['save_dir'], 'encD')
    kwargs_encD["component_name"] = "EncD"
    kwargs_encD["network_name"] = args["network_name"]
    kwargs_encD["kwargs_network"] = args["kwargs_encD"]
    kwargs_encD["kwargs_network"]["n_classes"] = args["n_classes"] + 1
    kwargs_encD["kwargs_network"]["n_latent_dim"] = args["n_latent_dim"]

    kwargs_encD["kwargs_optim"] = args["kwargs_encD_optim"]
    kwargs_encD["kwargs_optim"]["lr"] = args["lr_encD"]

    # Decoder discriminator
    kwargs_decD = kwargs_base.copy()
    kwargs_decD["save_path"] = "{}/{}.pth".format(args["save_dir"], "decD")
    # kwargs_enc['save_args_path'] = '{}/args_{}.json'.format(args['save_dir'], 'decD')
    kwargs_decD["component_name"] = "DecD"
    kwargs_decD["network_name"] = args["network_name"]
    kwargs_decD["kwargs_network"] = args["kwargs_decD"]
    kwargs_decD["kwargs_network"]["n_channels"] = len(args["channels"])
    kwargs_decD["kwargs_network"]["n_classes"] = args["n_classes"] + 1

    kwargs_decD["kwargs_optim"] = args["kwargs_decD_optim"]
    kwargs_decD["kwargs_optim"]["lr"] = args["lr_decD"]

    network_kwargs = {}
    network_kwargs["enc"] = kwargs_enc
    network_kwargs["dec"] = kwargs_dec

    if args["model_type"] == "ae":
        pass  # already setup!

    elif args["model_type"] == "aegan":
        network_kwargs["decD"] = kwargs_decD

    elif args["model_type"] == "aaegan":
        network_kwargs["decD"] = kwargs_decD
        network_kwargs["encD"] = kwargs_encD

    return network_kwargs


def setup_kwargs_trainer(args):
    kwargs_trainer = args["kwargs_model"]
    kwargs_trainer["n_epochs"] = args["nepochs"]
    kwargs_trainer["save_dir"] = args["save_dir"]
    kwargs_trainer["save_state_iter"] = args["saveStateIter"]
    kwargs_trainer["save_progress_iter"] = args["saveProgressIter"]
    kwargs_trainer["gpu_ids"] = args["gpu_ids"]

    return args["train_module"], kwargs_trainer


def setup_kwargs_loss(args, pt2):

    if "size_average" in args["kwargs_crit"]:
        args["kwargs_crit"]["size_average"] = bool(args["kwargs_crit"]["size_average"])

    kwargs_losses = {}
    kwargs_losses["crit_recon"] = {}
    kwargs_losses["crit_recon"]["name"] = args["crit_recon"]
    kwargs_losses["crit_recon"]["kwargs"] = args["kwargs_crit"]

    if pt2:
        kwargs_losses["crit_z_class"] = {}
        kwargs_losses["crit_z_class"]["name"] = args["crit_z_class"]
        kwargs_losses["crit_z_class"]["kwargs"] = args["kwargs_crit"]

        kwargs_losses["crit_z_ref"] = {}
        kwargs_losses["crit_z_ref"]["name"] = args["crit_z_ref"]
        kwargs_losses["crit_z_ref"]["kwargs"] = args["kwargs_crit"]

    return kwargs_losses


parser = argparse.ArgumentParser()

parser.add_argument("--gpu_ids", nargs="+", type=int, default=0, help="gpu id")
parser.add_argument("--myseed", type=int, default=0, help="random seed")
parser.add_argument(
    "--n_latent_dim", type=int, default=16, help="number of latent dimensions"
)

parser.add_argument(
    "--model_type",
    type=str,
    default="ae",
    help="model type; can be {'ae', 'aegan', 'aaegan'}",
)

parser.add_argument(
    "--lr_enc", type=float, default=0.0005, help="learning rate for encoder"
)
parser.add_argument(
    "--lr_dec", type=float, default=0.0005, help="learning rate for decoder"
)
parser.add_argument(
    "--lr_encD",
    type=float,
    default=0.0005,
    help="learning rate for encoder descriminator",
)
parser.add_argument(
    "--lr_decD",
    type=float,
    default=0.0005,
    help="learning rate for decoder descriminator",
)

parser.add_argument(
    "--kwargs_enc_optim",
    type=json.loads,
    default='{"betas": [0.5, 0.999]}',
    help="kwargs for encoder optimizer",
)
parser.add_argument(
    "--kwargs_dec_optim",
    type=json.loads,
    default='{"betas": [0.5, 0.999]}',
    help="kwargs for decoder optimizer",
)
parser.add_argument(
    "--kwargs_encD_optim",
    type=json.loads,
    default='{"betas": [0.5, 0.999]}',
    help="kwargs for encoder descriminator optimizer",
)
parser.add_argument(
    "--kwargs_decD_optim",
    type=json.loads,
    default='{"betas": [0.5, 0.999]}',
    help="kwargs for decoder descriminator optimizer",
)

parser.add_argument(
    "--kwargs_model", type=json.loads, default={}, help="kwargs for the model"
)

parser.add_argument(
    "--kwargs_enc", type=json.loads, default={}, help="kwargs for the enc"
)
parser.add_argument(
    "--kwargs_dec", type=json.loads, default={}, help="kwargs for the dec"
)
parser.add_argument(
    "--kwargs_encD",
    type=json.loads,
    default={},
    help="kwargs for the encoder descriminator",
)
parser.add_argument(
    "--kwargs_decD",
    type=json.loads,
    default={},
    help="kwargs for the decoder descriminator",
)

parser.add_argument(
    "--dataProvider", default="DataProvider", help="Dataprovider object"
)
parser.add_argument(
    "--kwargs_dp", type=json.loads, default={}, help="kwargs for the data provider"
)

parser.add_argument(
    "--crit_recon",
    default="torch.nn.BCELoss",
    help="Loss function for image reconstruction",
)
parser.add_argument(
    "--crit_z_class", default="torch.nn.NLLLoss", help="Loss function for class loss"
)
parser.add_argument(
    "--crit_z_ref", default="torch.nn.MSELoss", help="Loss function for reference loss"
)
parser.add_argument(
    "--kwargs_crit", type=json.loads, default={}, help="kwargs for loss functions"
)

parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--nepochs", type=int, default=250, help="total number of epochs")
parser.add_argument(
    "--nepochs_pt2", type=int, default=-1, help="total number of epochs"
)

parser.add_argument(
    "--network_name", default="waaegan", help="name of the model module"
)
parser.add_argument("--save_dir", type=str, default=None, help="save dir")
parser.add_argument(
    "--save_parent",
    type=str,
    default=None,
    help='parent save directory to save with autogenerated working directory (mutually exclusive to "--save_dir")',
)
parser.add_argument(
    "--saveProgressIter",
    type=int,
    default=1,
    help="number of iterations between saving progress",
)
parser.add_argument(
    "--saveStateIter",
    type=int,
    default=1,
    help="number of iterations between saving progress",
)

parser.add_argument("--data_save_path", default=None, help="save path of data file")
parser.add_argument(
    "--imdir",
    default="/root/data/release_4_1_17/results_v2/aligned/2D",
    help="location of images",
)

parser.add_argument("--ndat", type=int, default=-1, help="Number of data points to use")
parser.add_argument(
    "--optimizer", default="Adam", help="type of optimizer, can be {Adam, RMSprop}"
)

parser.add_argument("--train_module", default=None, help="training module")
parser.add_argument("--train_module_pt1", default=None, help="training module")
parser.add_argument("--train_module_pt2", default=None, help="training module")

parser.add_argument(
    "--channels_pt1",
    nargs="+",
    type=int,
    default=[0, 2],
    help="channels to use for part 1",
)
parser.add_argument(
    "--channels_pt2",
    nargs="+",
    type=int,
    default=[0, 1, 2],
    help="channels to use for part 2",
)

parser.add_argument(
    "--dtype",
    default="float",
    help="data type that the dataprovider uses. Only 'float' supported.",
)

parser.add_argument(
    "--overwrite_opts", default=False, type=str2bool, help="Overwrite options file"
)
parser.add_argument("--skip_pt1", default=False, type=str2bool, help="Skip pt 1")

parser.add_argument(
    "--ref_dir",
    default="ref_model",
    type=str,
    help="Directory name for reference model",
)
parser.add_argument(
    "--struct_dir",
    default="struct_model",
    type=str,
    help="Directory name for structure model",
)

parser.add_argument(
    "--pt2_use_pretrained",
    default=False,
    type=str2bool,
    help="User part 1 models to start part 2",
)

args = vars(parser.parse_args())

args = setup(args)

save_dir = args["save_dir"]

# load the all of the parameters
args = utils.save_load_dict(
    "{}/args.json".format(save_dir), args, args["overwrite_opts"]
)


args["save_dir"] = "{}/{}".format(save_dir, args["ref_dir"])
save_dir_pt1 = args["save_dir"]

if not os.path.exists(args["save_dir"]):
    os.makedirs(args["save_dir"])

# load the dataprovider
args["channel_inds"] = args["channels_pt1"]
dp_name, dp_kwargs = utils.save_load_dict(
    "{}/args_dp.json".format(args["save_dir"]),
    setup_kwargs_data_provider(args),
    args["overwrite_opts"],
)
dp = model_utils.load_data_provider(dp_name, **dp_kwargs)

# load the trainer model
trainer_name, trainer_kwargs = utils.save_load_dict(
    "{}/args_trainer.json".format(args["save_dir"]),
    setup_kwargs_trainer(args),
    args["overwrite_opts"],
)

trainer_module = importlib.import_module(
    "integrated_cell.models.{}".format(trainer_name)
)

# load the networks
args["n_classes"] = 0
args["n_ref"] = 0
args["channels"] = args["channels_pt1"]

net_kwargs = setup_kwargs_network(args)

networks, optimizers = {}, {}
for net_name in net_kwargs:
    args_save_path = "{}/args_{}.json".format(args["save_dir"], net_name)
    net_kwargs[net_name] = utils.save_load_dict(
        args_save_path, net_kwargs[net_name], args["overwrite_opts"]
    )

    networks[net_name], optimizers["opt_{}".format(net_name)] = utils.load_network(
        **net_kwargs[net_name]
    )


# load the loss functions
loss_info = setup_kwargs_loss(args, pt2=False)

losses = {}
for k in loss_info:
    losses[k] = model_utils.load_loss(loss_info[k]["name"], loss_info[k]["kwargs"])

#######
# TRAIN REFERENCE MODEL
#######
if not os.path.exists(args["save_dir"]):
    os.makedirs(args["save_dir"])

print(args)
model = trainer_module.Model(
    data_provider=dp, **networks, **optimizers, **losses, **trainer_kwargs
)

model.train()

#######
# DONE TRAINING REFERENCE MODEL
#######

#######
# TRAIN STRUCTURE MODEL
#######

embeddings_path = "{}/embeddings.pkl".format(args["save_dir"])
embeddings = model_utils.load_embeddings(embeddings_path, model.enc, dp)

models = None
optimizers = None

dp.embeddings = embeddings
dp.channelInds = args["channels_pt2"]

# load the networks
args["n_classes"] = dp.get_n_classes()
args["n_ref"] = dp.get_n_ref()
args["channels"] = args["channels_pt2"]

args["save_dir"] = "{}/{}".format(save_dir, args["struct_dir"])
if not os.path.exists(args["save_dir"]):
    os.makedirs(args["save_dir"])

# load the trainer model
trainer_name, trainer_kwargs = utils.save_load_dict(
    "{}/args_trainer.json".format(args["save_dir"]),
    setup_kwargs_trainer(args),
    args["overwrite_opts"],
)

trainer_module = importlib.import_module(
    "integrated_cell.models.{}".format(trainer_name)
)


net_kwargs = setup_kwargs_network(args)

networks, optimizers = {}, {}
for net_name in net_kwargs:

    if args["pt2_use_pretrained"]:
        net_kwargs[net_name]["kwargs_network"][
            "pretrained_path"
        ] = "{}/args_{}.json".format(save_dir_pt1, net_name)

    args_save_path = "{}/args_{}.json".format(args["save_dir"], net_name)
    net_kwargs[net_name] = utils.save_load_dict(
        args_save_path, net_kwargs[net_name], args["overwrite_opts"]
    )

    networks[net_name], optimizers["opt_{}".format(net_name)] = utils.load_network(
        **net_kwargs[net_name]
    )

# load the loss functions
loss_info = setup_kwargs_loss(args, pt2=True)

losses = {}
for k in loss_info:
    losses[k] = model_utils.load_loss(loss_info[k]["name"], loss_info[k]["kwargs"])

#######
# TRAIN REFERENCE MODEL
#######
if not os.path.exists(args["save_dir"]):
    os.makedirs(args["save_dir"])

print(args)
model = trainer_module.Model(
    data_provider=dp, **networks, **optimizers, **losses, **trainer_kwargs
)

model.train()

print("Finished Training")

embeddings_path = "{}/embeddings.pkl".format(args["save_dir"])
embeddings = model_utils.load_embeddings(embeddings_path, model.enc, dp)

#######
# DONE TRAINING STRUCTURE MODEL
#######
