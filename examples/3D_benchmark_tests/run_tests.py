import argparse
import os
import subprocess
import hashlib
import pickle
import numpy as np
import itertools

import matplotlib

matplotlib.use("Agg")  # noqa

import matplotlib.pyplot as plt  # noqa
from matplotlib.lines import Line2D  # noqa
from matplotlib import cm  # noqa


# This is a script to benchmark how GPU scaling effects the iteration time of 3D IC models


def main(use_current_results=False, save_dir="./"):

    experiments = get_experiments(save_dir)

    if not use_current_results:
        run_experiments(experiments)

    experiments = get_experiment_info(experiments)

    plot_experiments(experiments, save_dir)

    experiments_tmp = list()
    for experiment in experiments:
        if len(experiment["gpu_id"]) == 8:
            experiments_tmp.append(experiment)

    plot_experiments(experiments_tmp, save_dir, save_name="stats_8gpu.png")


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


def get_experiment_info(experiments):
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

    return experiments


def plot_experiments(experiments, save_dir, save_name="stats.png"):
    # plot the results

    docker_indicator = [
        "D, " if experiment["function_call"] == "bash run_docker.sh" else ""
        for experiment in experiments
    ]
    apex_indicator = [
        "A, " if experiment["trainer_type"] == "cbvae_apex" else ""
        for experiment in experiments
    ]

    gpu_ids = [experiment["str_gpu_id"] for experiment in experiments]

    exp_strings = [
        "".join(exp) for exp in zip(docker_indicator, apex_indicator, gpu_ids)
    ]

    experiments = np.array(experiments)

    u_exps, exp_inds = np.unique(exp_strings, return_inverse=True)

    colors = cm.hsv((255 * np.arange(len(u_exps)) / len(u_exps)).astype("uint8"))

    plt.figure(figsize=[8, 4])

    for i in range(len(u_exps)):
        exp_ind = exp_inds == i

        exps_tmp = experiments[exp_ind]

        mu = np.array([exp["iter_time_mu"] for exp in exps_tmp])
        std = np.array([exp["iter_time_std"] for exp in exps_tmp])
        batch_size = np.array([exp["batch_size"] for exp in exps_tmp])

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

    batch_sizes = np.unique([experiment["batch_size"] for experiment in experiments])

    batch_sizes = batch_sizes[batch_sizes < plt.xlim()[1]]

    plt.xticks(batch_sizes)

    plt.xlabel("batch size")
    plt.ylabel("avg iter time (s)")

    lgd = plt.legend(
        handles=legend_elements, title="Experiments", bbox_to_anchor=(1, 1)
    )

    plt.savefig(
        "{}/{}".format(save_dir, save_name),
        bbox_extra_artists=[lgd],
        dpi=90,
        bbox_inches="tight",
    )
    plt.close()


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
