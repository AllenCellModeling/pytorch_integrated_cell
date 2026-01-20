import datetime
import os
import json
import importlib
import warnings
import argparse
import shutil
import hashlib
import pickle

import numpy as np
import pandas as pd
import torch

from ..model_utils import load_state as load_state
from .. import layers


# conditional beta variational autencoder
def reparameterize(mu, log_var, add_noise=True):
    if add_noise:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        out = mu + eps * std
    else:
        out = mu

    return out


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


def load_losses(args):
    # load the loss functions
    kwargs_losses = {}
    kwargs_losses["crit_recon"] = {}
    kwargs_losses["crit_recon"]["name"] = args["crit_recon"]
    kwargs_losses["crit_recon"]["kwargs"] = args["kwargs_crit_recon"]

    if args["model_type"] == "ae":
        pass  # already setup!

    elif args["model_type"] == "aae":
        kwargs_losses["crit_encD"] = {}
        kwargs_losses["crit_encD"]["name"] = args["crit_encD"]
        kwargs_losses["crit_encD"]["kwargs"] = args["kwargs_crit_encD"]

    elif args["model_type"] == "aegan":
        kwargs_losses["crit_decD"] = {}
        kwargs_losses["crit_decD"]["name"] = args["crit_decD"]
        kwargs_losses["crit_decD"]["kwargs"] = args["kwargs_crit_decD"]

    elif args["model_type"] == "aaegan":
        kwargs_losses["crit_decD"] = {}
        kwargs_losses["crit_decD"]["name"] = args["crit_decD"]
        kwargs_losses["crit_decD"]["kwargs"] = args["kwargs_crit_decD"]

        kwargs_losses["crit_encD"] = {}
        kwargs_losses["crit_encD"]["name"] = args["crit_encD"]
        kwargs_losses["crit_encD"]["kwargs"] = args["kwargs_crit_encD"]

    losses = {}
    for k in kwargs_losses:
        losses[k] = load_object(kwargs_losses[k]["name"], kwargs_losses[k]["kwargs"])

    return losses


def load_object(object_name, object_kwargs):
    object_module, object_name = object_name.rsplit(".", 1)
    object_module = importlib.import_module(object_module)

    return getattr(object_module, object_name)(**object_kwargs)


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


def load_network(
    network_name,
    component_name,
    kwargs_network,
    optim_name,
    kwargs_optim,
    save_path,
    pretrained_path=None,
    pretrained_reset_optim=True,
    gpu_ids=None,
    init_meth="normal",
    verbose=True,
):

    model_provider = importlib.import_module("integrated_cell.networks." + network_name)

    # overwrite existing gpu ids
    if gpu_ids is not None:
        kwargs_network["gpu_ids"] = gpu_ids

    network = getattr(model_provider, component_name)(**kwargs_network)

    def w_init(x):
        weights_init(x, init_meth)

    network.apply(w_init)
    network.cuda(gpu_ids[0])

    kwargs_optim["params"] = network.parameters()

    if optim_name.lower() == "adam":
        optim_name = "torch.optim.Adam"

    optimizer = load_object(optim_name, kwargs_optim)

    if os.path.exists(save_path):
        # if the save path exists
        if verbose:
            print("loading from {}".format(save_path))
        load_state(network, optimizer, save_path, gpu_ids[0])

    elif pretrained_path is not None:
        # otherwise try to load from a pretrained path
        if verbose:
            print(
                "loading pretrained {}.{} from {}".format(
                    network_name, component_name, pretrained_path
                )
            )
        load_state(network, optimizer, pretrained_path, gpu_ids[0])

        if pretrained_reset_optim:
            kwargs_optim["params"] = network.parameters()

            if optim_name.lower() == "adam":
                optim_name = "torch.optim.Adam"

            optimizer = load_object(optim_name, kwargs_optim)

    return network, optimizer


def load_network_from_dir(
    model_save_dir,
    parent_dir="./",
    net_names=["enc", "dec"],
    suffix="",
    gpu_ids=[0],
    load_dataprovider=True,
):

    if suffix is None:
        suffix = ""

    args_file = "{}/args.json".format(model_save_dir)

    with open(args_file, "r") as f:
        args = json.load(f)

    args["save_dir"] = "{}/{}".format(model_save_dir, args["ref_dir"])

    if load_dataprovider:
        dp_name, dp_kwargs = save_load_dict("{}/args_dp.json".format(args["save_dir"]))
        dp_kwargs["save_path"] = dp_kwargs["save_path"].replace("./", parent_dir)
        dp = load_data_provider(dp_name, **dp_kwargs)
    else:
        dp = None

    net_kwargs = {}
    networks = {}

    for net_name in net_names:
        args_save_path = "{}/args_{}.json".format(args["save_dir"], net_name)

        net_kwargs[net_name] = save_load_dict(args_save_path)
        net_kwargs[net_name]["save_path"] = net_kwargs[net_name]["save_path"].replace(
            "./", parent_dir
        )

        net_kwargs[net_name]["save_path"] = (
            net_kwargs[net_name]["save_path"][:-4]
            + suffix
            + net_kwargs[net_name]["save_path"][-4:]
        )

        networks[net_name], _ = load_network(**net_kwargs[net_name])

    return networks, dp, args


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

    elif activation.lower() == "channelsoftmax":
        return layers.ChannelSoftmax()

    elif activation.lower() == "softplus":
        return torch.nn.Softplus()


def str2rand(strings, seed=0):
    # prepend the 'seed' to the unique file path, then hash with SHA512
    salted_string = [str(seed) + str(string) for string in strings]
    hash_strings = [
        hashlib.sha512(string.encode("utf-8")).hexdigest() for string in salted_string
    ]

    rand_nums = list()
    # Pull out the first 5 digits to get a value between 0-1 inclusive
    for hash_string in hash_strings:
        str_nums = [char for pos, char in enumerate(hash_string) if char.isdigit()]
        str_num = "".join(str_nums[0:5])
        num = float(str_num) / 100000
        rand_nums.append(num)

    rand_nums = np.array(rand_nums)

    return rand_nums


def sample_image(dec, n_imgs=1, classes_to_generate=None):
    # classes_to_generate is a list of integers corresponding to classes to generate

    n_classes = dec.n_classes
    n_ref_dim = dec.n_ref
    n_latent_dim = dec.n_latent_dim

    if classes_to_generate is None:
        classes_to_generate = torch.tensor(np.arange(0, n_classes))

    n_classes_to_generate = len(classes_to_generate)

    vec_ref = torch.zeros(1, n_ref_dim).float().cuda()
    vec_ref.normal_()

    vec_ref.repeat([n_classes_to_generate, 1])

    vec_struct = torch.zeros(n_classes_to_generate, n_latent_dim).float().cuda()
    vec_struct.normal_()

    vec_class = torch.zeros(n_classes_to_generate, n_classes).float().cuda()
    for i in range(n_classes_to_generate):
        vec_class[i, classes_to_generate[i]] = 1

    with torch.no_grad():
        img_out = dec([vec_class, vec_ref, vec_struct]).detach().cpu()

    return (
        img_out,
        [vec_class.detach().cpu(), vec_ref.detach().cpu(), vec_struct.detach().cpu()],
    )


def autoencode_image(enc, dec, im_in, im_class):
    n_classes = dec.n_classes

    im_in = im_in.cuda()

    vec_class = torch.zeros(1, n_classes).float().cuda()
    vec_class[0, im_class] = 1

    with torch.no_grad():
        zAll = enc(im_in, vec_class)

    for j in range(len(zAll)):
        zAll[j] = zAll[j][0]

    with torch.no_grad():
        xHat = dec([vec_class] + zAll)

    return xHat.detach().cpu()


def predict_image(enc, dec, im_in, im_class):
    n_classes = dec.n_classes

    im_in = im_in.cuda()

    vec_class = torch.zeros(1, n_classes).float().cuda()
    vec_class[0, im_class] = 1

    with torch.no_grad():
        zAll = enc(im_in, vec_class)

    zAll[0] = zAll[0][0]
    zAll[1] = torch.zeros(zAll[1][0].shape).float().cuda()

    with torch.no_grad():
        xHat = dec([vec_class] + zAll)

    return xHat.detach().cpu()


def save_load_mitosis_annotations(data_provider, save_dir=None):
    # queries labkey and populates the data_provider with mitosis annotations that correspond to the cell indecies
    if save_dir is None:
        save_dir = data_provider.image_parent

    save_path = "{}/mitosis_annotations.pkl".format(save_dir)

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            mito_data = pickle.load(f)
    else:
        # get mito data from labkey and split it into two groups
        # binary mitosis labels and resolved (m1, m2, etc) labels

        from lkaccess import LabKey

        lk = LabKey(host="aics")
        mito_data = lk.select_rows_as_list(
            schema_name="processing",
            query_name="MitoticAnnotation",
            sort="MitoticAnnotation",
            columns=["CellId", "MitoticStateId", "MitoticStateId/Name", "Complete"],
        )

        mito_data = pd.DataFrame(mito_data)

        with open(save_path, "wb") as f:
            pickle.dump(mito_data, f)

    mito_binary_inds = mito_data["MitoticStateId/Name"] == "Mitosis"
    not_mito_inds = mito_data["MitoticStateId/Name"] == "M0"

    mito_data_binary = mito_data[mito_binary_inds | not_mito_inds]
    mito_data_resolved = mito_data[~mito_binary_inds]

    # print(np.unique(mito_data["MitoticStateId/Name"]))

    modes = [k for k in data_provider.data]

    for mode in modes:
        mito_states = list()
        for cellId in data_provider.data[mode]["CellId"]:
            mito_state = mito_data_binary["MitoticStateId/Name"][
                mito_data_binary["CellId"] == cellId
            ].values
            if len(mito_state) == 0:
                mito_state = "unknown"

            mito_states.append(mito_state[0])

        data_provider.data[mode]["mito_state_binary"] = np.array(mito_states)
        data_provider.data[mode]["mito_state_binary_ind"] = np.array(
            np.unique(mito_states, return_inverse=True)[1]
        )

    for mode in modes:
        mito_states = list()
        for cellId in data_provider.data[mode]["CellId"]:
            mito_state = mito_data_resolved["MitoticStateId/Name"][
                mito_data_resolved["CellId"] == cellId
            ].values
            if len(mito_state) == 0:
                mito_state = "u"

            mito_states.append(mito_state[0])

        data_provider.data[mode]["mito_state_resolved"] = np.array(mito_states)
        data_provider.data[mode]["mito_state_resolved_ind"] = np.array(
            np.unique(mito_states, return_inverse=True)[1]
        )

    mito_state_names = np.unique(
        data_provider.data[mode]["mito_state_binary"][
            data_provider.data[mode]["mito_state_binary"] != "u"
        ]
    )
    mito_state_names = [name.replace("M0", "Interphase") for name in mito_state_names]

    data_provider.mito_state_names = mito_state_names

    return data_provider


def load_data_provider(
    module_name, save_path, batch_size, im_dir, channel_names=None, n_dat=-1, **kwargs_dp
):
    DP = importlib.import_module("integrated_cell.data_providers." + module_name)

    if os.path.exists(save_path):
        dp = pickle.load(open(save_path, "rb"))
        dp.image_parent = im_dir
    else:
        dp = DP.DataProvider(
            image_parent=im_dir, batch_size=batch_size, n_dat=n_dat, **kwargs_dp
        )
        pickle.dump(dp, open(save_path, "wb"))

    if not hasattr(dp, "normalize_intensity"):
        dp.normalize_intensity = False

    dp.batch_size = batch_size
    dp.set_n_dat(n_dat, "train")

    if channel_names is not None:
        dp.channel_names = channel_names

    for k in dp.data:
        dp.data[k]["CellId"] = dp.csv_data["CellId"].values[dp.data[k]["inds"]]

    try:
        save_load_mitosis_annotations(dp)
    except ModuleNotFoundError:
        warnings.warn(
            'Could not load mitosis annotations. Probably because you dont have "lkaccess" (for internal AICS use only).'
        )

    return dp


def load_drug_data_provider(
    data_provider,
    args,
    image_parent="/allen/aics/modeling/gregj/results/ipp/scp_drug_pilot_fixed/",
):

    image_parent = "/allen/aics/modeling/gregj/results/ipp/scp_drug_pilot_fixed/"

    data_jobs_out_path = "{}/{}".format(image_parent, "data_jobs_out.csv")

    data_jobs_out_corrected_path = "{}/{}".format(
        image_parent, "data_jobs_out_corrected.csv"
    )

    if not os.path.exists(data_jobs_out_corrected_path):
        df = pd.read_csv(data_jobs_out_path)

        # These are the default values from the single cell processing pipeline
        df["ch_memb"] = 3
        df["ch_struct"] = 4
        df["ch_dna"] = 2
        df["ch_seg_cell"] = 1
        df["ch_seg_nuc"] = 0
        df["ch_trans"] = 5

        df.to_csv(data_jobs_out_corrected_path)

    kwargs_dp_drugs = args["kwargs_dp"]

    kwargs_dp_drugs["csv_name"] = "data_jobs_out_corrected.csv"
    kwargs_dp_drugs["hold_out"] = 1
    kwargs_dp_drugs["batch_size"] = 1
    kwargs_dp_drugs["image_parent"] = image_parent

    dp_drugs = load_object(
        "integrated_cell.data_providers.{}.DataProvider".format(args["dataProvider"]),
        kwargs_dp_drugs,
    )

    dp_drugs.csv_data["drug_label"][
        dp_drugs.csv_data["drug_label"] == "s-Nitro-Blebbistatin"
    ] = "S-Nitro-Blebbistatin"
    experiment_info = dp_drugs.csv_data[["concentration", "drug_label"]]

    u_conc = np.unique(experiment_info["concentration"])
    u_drug = np.unique(experiment_info["drug_label"])

    dp_drugs.csv_data["drug_name"] = ""

    for i, conc in enumerate(u_conc):
        for j, drug in enumerate(u_drug):
            experiment_inds = np.all(
                np.vstack(
                    [
                        experiment_info["concentration"] == conc,
                        experiment_info["drug_label"] == drug,
                    ]
                ),
                axis=0,
            )
            if np.any(experiment_inds):
                dp_drugs.csv_data["drug_label"][experiment_inds] = r"{} {} μM".format(
                    drug, conc
                )
                dp_drugs.csv_data["drug_name"][experiment_inds] = drug

    # Do some surgery on the drug data provider to make sure it has the same classes in the same order
    drug_structures = dp_drugs.label_names
    complete_structures = data_provider.label_names

    drug_label_dict = {}
    for structure in drug_structures:
        drug_label_dict[structure] = np.where(complete_structures == structure)[0]

    # set the structure labels
    dp_drugs.labels = np.hstack(
        [drug_label_dict[structure] for structure in dp_drugs.image_classes]
    )

    nimgs = len(dp_drugs.labels)

    onehot = np.zeros((nimgs, np.max(data_provider.labels) + 1))
    onehot[np.arange(nimgs), dp_drugs.labels] = 1

    dp_drugs.labels_onehot = onehot

    experiment_info = dp_drugs.csv_data[["concentration", "drug_label"]]

    drug_names, drug_ids = np.unique(experiment_info["drug_label"], return_inverse=True)
    drug_names = np.hstack([["Control"], drug_names])
    drug_ids = drug_ids + 1  # reserve 0 for "control"

    print("DrugID-Concentration combinations")
    print(
        np.unique(
            np.vstack([drug_ids, experiment_info["concentration"].values]).transpose(),
            axis=0,
        )
    )

    # set the drug IDs as the reference coordinate
    drug_info = {}
    for k in dp_drugs.data:
        drug_info[k] = drug_ids[np.array(dp_drugs.data[k]["inds"])]

    dp_drugs.drug_info = drug_info
    dp_drugs.drug_names = drug_names
    # dp_drugs.drug_ids = drug_ids

    # dp_drugs.set_ref(drug_info)

    return dp_drugs
