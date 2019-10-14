import argparse
import os
import subprocess
import hashlib
import pickle
import numpy as np
import itertools
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # noqa

import matplotlib.pyplot as plt  # noqa
from matplotlib.lines import Line2D  # noqa
from matplotlib import cm  # noqa


# This is a script to benchmark how GPU scaling effects the iteration time of 3D IC models


def get_experiments(save_parent):

    experiment_dict = {}
    experiment_dict["function_call"] = ["bash run_docker.sh", "bash run_3D.sh"]
    experiment_dict["trainer_type"] = ["cbvae_apex", "cbvae"]
    experiment_dict["gpu_id"] = [
        [2],
        [2, 3],
        [3, 4],
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]
    experiment_dict["batch_size"] = [8, 16, 32, 64, 128, 256]

    # total amount of data to use
    n_iter = 11

    param_names = [k for k in experiment_dict]
    param_values = [experiment_dict[k] for k in experiment_dict]

    # build up the list of experiments
    experiments = list()

    for param_permutation in itertools.product(*param_values):

        experiment = dict(zip(param_names, param_permutation))

        experiment["n_dat"] = int(n_iter * experiment["batch_size"])

        str_gpu_id = " ".join([str(gpu) for gpu in experiment["gpu_id"]])
        experiment["str_gpu_id"] = str_gpu_id

        # make a big string with all the crap in the dictionary and hash it, so we can get our unique directory for the experiment
        mystr = "_".join([str(experiment[k]) for k in experiment])
        mystr2hash = hashlib.sha1(mystr.encode()).hexdigest()

        # specify a save dir with that hash
        save_dir = "{}/results/test_{}".format(save_parent, mystr2hash)

        experiment["save_dir"] = save_dir

        # make the executable string
        exe_str = "{function_call} '{str_gpu_id}' {save_dir} {trainer_type} {batch_size} {n_dat}".format(
            **experiment
        )

        experiment["exe_str"] = exe_str

        experiments.append(experiment)

    return experiments


def run_experiments(experiments):

    # execute each experiment
    for experiment in experiments:
        print(experiment["exe_str"])
        if not os.path.exists(experiment["save_dir"]):
            subprocess.call(experiment["exe_str"], shell=True)

    return


def get_experiment_info_dataframe(experiments):
    # looks in the model directory and pulls out relevant info
    for experiment in experiments:

        logger_path = "{}/ref_model/logger_tmp.pkl".format(experiment["save_dir"])

        if os.path.exists(experiment["save_dir"]):
            if os.path.exists(logger_path):
                logger = pickle.load(open(logger_path, "rb"))

                experiment["iter_time_mu"] = np.mean(np.array(logger.log["time"][1:]))
                experiment["iter_time_std"] = np.std(
                    np.array(logger.log["time"][1:])
                ) / np.sqrt(len(np.array(logger.log["time"][1:])))
            else:
                # model failure (e.g. ran out of memory)
                experiment["iter_time_mu"] = 0
                experiment["iter_time_std"] = 0
        else:
            # this model was never ran
            experiment["iter_time_mu"] = -1
            experiment["iter_time_std"] = -1

    df_experiments = pd.DataFrame(experiments)

    return df_experiments


def plot_experiments(experiments, save_path):
    # plot the results

    docker_indicator = [
        "D, " if function_call == "bash run_docker.sh" else ""
        for function_call in experiments["function_call"]
    ]
    apex_indicator = [
        "A, " if trainer_type == "cbvae_apex" else ""
        for trainer_type in experiments["trainer_type"]
    ]

    gpu_ids = experiments["str_gpu_id"]

    exp_strings = [
        "".join(exp) for exp in zip(docker_indicator, apex_indicator, gpu_ids)
    ]

    u_exps, exp_inds = np.unique(exp_strings, return_inverse=True)

    colors = cm.hsv((255 * np.arange(len(u_exps)) / len(u_exps)).astype("uint8"))

    plt.figure(figsize=[8, 4])

    for i in range(len(u_exps)):
        exp_ind = exp_inds == i

        exps_tmp = experiments.iloc[exp_ind]

        mu = exps_tmp["iter_time_mu"].values
        std = exps_tmp["iter_time_std"].values
        batch_size = exps_tmp["batch_size"].values

        # discard 0 time iterations
        keep_inds = mu > 0

        mu = mu[keep_inds]
        std = std[keep_inds]
        batch_size = batch_size[keep_inds]

        # sort by batch size
        order = np.argsort(batch_size)

        mu = mu[order]
        std = std[order]
        batch_size = batch_size[order]

        plt.plot(batch_size, mu, color=colors[i])
        plt.plot(batch_size, mu + std, color=colors[i], linestyle="--")
        plt.plot(batch_size, mu - std, color=colors[i], linestyle="--")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(u_exp),
            markerfacecolor=colors[i],
            markersize=5,
        )
        for i, u_exp in enumerate(u_exps)
    ]

    batch_sizes = np.unique(experiments["batch_size"])

    batch_sizes = batch_sizes[batch_sizes < plt.xlim()[1]]

    plt.xticks(batch_sizes)

    plt.xlabel("batch size")
    plt.ylabel("avg iter time (s)")

    lgd = plt.legend(
        handles=legend_elements, title="Experiments", bbox_to_anchor=(1, 1)
    )

    plt.savefig(save_path, bbox_extra_artists=[lgd], dpi=90, bbox_inches="tight")
    plt.close()


def pair_scatter(experiments, pair_on, save_path, markers=["*", "o"]):
    gpu_ids = np.unique(experiments["str_gpu_id"])
    pair_vals = np.unique(experiments[pair_on])

    if len(pair_vals) != 2:
        raise Exception("The pair_on parameter is invalid")

    colors = cm.hsv((255 * np.arange(len(gpu_ids)) / len(gpu_ids)).astype("uint8"))

    plt.figure()

    for i, gpu_id in enumerate(gpu_ids):
        experiment_inds = experiments["str_gpu_id"] == gpu_id
        experiments_tmp = experiments[experiment_inds]

        exps = list()
        for pair_val in pair_vals:
            exp = experiments_tmp[experiments_tmp[pair_on] == pair_val]

            exp = exp.iloc[np.argmax(exp["iter_time_mu"].values)]
            exps.append(exp)

        exps = pd.DataFrame(exps)

        plt.plot(
            exps["batch_size"],
            exps["iter_time_mu"],
            color=colors[i],
            label="GPU IDs: " + gpu_id,
        )

        for pair in range(2):

            label = None
            if i == 0:
                label = pair_vals[pair]

            plt.scatter(
                exps["batch_size"].iloc[pair],
                exps["iter_time_mu"].iloc[pair],
                color="k",
                marker=markers[pair],
                label=label,
            )

    plt.legend(bbox_to_anchor=(1, 1))

    plt.xlabel("batch size")
    plt.ylabel("avg iter time (s)")

    ylim = list(plt.ylim())
    ylim[0] = 0
    plt.ylim(ylim)

    batch_sizes = np.unique(experiments["batch_size"])
    batch_sizes = batch_sizes[batch_sizes < plt.xlim()[1]]
    plt.xticks(batch_sizes)

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(use_current_results=False, save_dir="./"):

    experiments = get_experiments(save_dir)

    if not use_current_results:
        run_experiments(experiments)

    df_experiments = get_experiment_info_dataframe(experiments)

    plots_dir = "{}/plots".format(save_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # ###
    # Plots
    # ###

    # plot everything
    plot_experiments(
        df_experiments, save_path="{}/{}".format(plots_dir, "stats_all.png")
    )

    # 8gpu experiments
    df_experiments_tmp = df_experiments.iloc[
        np.array([len(gpu_ids) for gpu_ids in df_experiments["gpu_id"]]) == 8
    ]
    plot_experiments(
        df_experiments_tmp, save_path="{}/{}".format(plots_dir, "stats_8gpu.png")
    )

    is_apex = np.array(
        ["apex" in trainer_type for trainer_type in df_experiments.trainer_type]
    )
    is_docker = np.array(
        ["docker" in function_call for function_call in df_experiments.function_call]
    )

    # Docker vs Non-Docker experiments (no Apex)
    pair_scatter(
        df_experiments.iloc[~is_apex],
        pair_on="function_call",
        save_path="{}/{}".format(plots_dir, "stats_docker_vs_not_docker.png"),
    )

    # Apex vs Non-Apex experiments (Docker only)
    pair_scatter(
        df_experiments.iloc[is_docker],
        pair_on="trainer_type",
        save_path="{}/{}".format(plots_dir, "stats_apex_vs_not_apex.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_current_results",
        default=False,
        type=str2bool,
        help="Continue printing results without running models",
    )
    parser.add_argument(
        "--save_dir", default="./", type=str, help="Save directory for results"
    )

    args = vars(parser.parse_args())

    main(use_current_results=args["use_current_results"], save_dir=args["save_dir"])
