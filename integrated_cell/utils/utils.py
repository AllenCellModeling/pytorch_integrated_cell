import torch
import datetime
import os
import json
import importlib
from .. import model_utils
import warnings
import argparse
import shutil


def index_to_onehot(index, n_classes):
    index = index.long().unsqueeze(1)

    onehot = torch.zeros(len(index), n_classes).type_as(index).float()
    onehot.scatter_(1, index, 1)

    return onehot


def memReport():
    import gc

    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())


def cpuStats():
    import psutil
    import sys

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB...I think
    print("memory GB:", memoryUse)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_load_dict(save_path, args=None, overwrite=False, verbose=True):
    # saves a dictionary, 'args', as a json file. Or loads if it exists.

    if os.path.exists(save_path) and not overwrite:
        warnings.warn(
            "args file exists and overwrite is not set to True. Using existing args file."
        )

        # load argsions file
        with open(save_path, "rb") as f:
            args = json.load(f)
    else:
        # make a copy if the args file exists
        if os.path.exists(save_path):
            the_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            shutil.copyfile(save_path, "{0}_{1}".format(save_path, the_time))

        with open(save_path, "w") as f:
            json.dump(args, f, indent=4, sort_keys=True)

    return args


def load_network(
    network_name,
    component_name,
    kwargs_network,
    optim_name,
    kwargs_optim,
    save_path,
    gpu_ids,
    init_meth="normal",
    verbose=True,
):

    model_provider = importlib.import_module("integrated_cell.networks." + network_name)

    if "gpu_ids" not in kwargs_network:
        kwargs_network["gpu_ids"] = gpu_ids

    network = getattr(model_provider, component_name)(**kwargs_network)

    def w_init(x):
        weights_init(x, init_meth)

    network.apply(w_init)
    network.cuda(gpu_ids[0])

    optimizer_provider = importlib.import_module("torch.optim")
    optimizer = getattr(optimizer_provider, optim_name)(
        params=network.parameters(), **kwargs_optim
    )

    if os.path.exists(save_path):
        if verbose:
            print("loading from {}".format(save_path))
        model_utils.load_state(network, optimizer, save_path, gpu_ids[0])

    return network, optimizer


# def load_results_from_dir(results_dir):

#     return dp, models


def weights_init(m, init_meth="normal"):
    classname = m.__class__.__name__

    if init_meth == "normal":
        if classname.find("Conv") != -1:
            try:
                m.weight.data.normal_(0.0, 0.02)
            except:  # noqa
                pass
        elif classname.find("BatchNorm") != -1:
            try:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            except:  # noqa
                pass

    elif init_meth == "ortho":
        if classname.find("Conv") != -1:
            try:
                torch.nn.init.orthogonal_(m.weight.data)
            except:  # noqa
                pass
        elif classname.find("BatchNorm") != -1:
            try:
                torch.nn.init.orthogonal_(m.weight.data)
                m.bias.data.fill_(0)
            except:  # noqa
                pass


def load_network_from_args_path(args_path):
    network_args = save_load_dict(args_path, None, False)
    network, optimizer = load_network(**network_args)

    return network, optimizer, network_args


def get_activation(activation):
    if activation is None or activation.lower() == "none":
        return torch.nn.Sequential()

    elif activation.lower() == "relu":
        return torch.nn.ReLU(inplace=True)

    elif activation.lower() == "prelu":
        return torch.nn.PReLU()

    elif activation.lower() == "sigmoid":
        return torch.nn.Sigmoid()

    elif activation.lower() == "leakyrelu":
        return torch.nn.LeakyReLU(0.2, inplace=True)
