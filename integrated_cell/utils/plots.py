import numpy as np
import matplotlib as mpl
from matplotlib.ticker import NullFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle

mpl.use("Agg")  # noqa


dpi = 100
figx = 6
figy = 4.5


def history(logger, save_path):

    # Figure out the default color order, and use these for the plots
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(figx, figy), dpi=dpi)

    ax = plt.gca()

    plts = list()

    # Plot reconstruction loss
    plts += ax.plot(
        logger.log["iter"], logger.log["reconLoss"], label="reconLoss", color=colors[0]
    )

    plt.ylabel("reconLoss")

    ax_max = np.percentile(logger.log["reconLoss"], 99)
    ax_min = np.percentile(logger.log["reconLoss"], 0)

    ax.set_ylim([ax_min, ax_max])

    # Plot everything else that isn't below
    do_not_print = ["epoch", "iter", "time", "reconLoss"]

    # Print off the reconLoss on it's own scale
    ax2 = plt.gca().twinx()

    y_vals = list()

    i = 1
    for field in logger.fields:
        if field not in do_not_print:
            plts += ax2.plot(
                logger.log["iter"], logger.log[field], label=field, color=colors[i]
            )
            y_vals += logger.log[field]
            i += 1

    if len(y_vals) > 0:
        ax_max = np.percentile(np.hstack(y_vals), 99.5)
        ax_min = np.percentile(np.hstack(y_vals), 0)

        ax2.set_ylim([ax_min, ax_max])

    # Get all the labels for the legend from both axes
    labs = [l.get_label() for l in plts]

    # Print legend
    ax.legend(plts, labs)

    plt.ylabel("loss")
    plt.title("History")
    plt.xlabel("iteration")

    # Save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def short_history(logger, save_path, max_history_len=10000):
    history = int(len(logger.log["epoch"]) / 2)

    if history > max_history_len:
        history = max_history_len

    x = logger.log["iter"][-history:]
    y = logger.log["reconLoss"][-history:]

    epochs = np.floor(np.array(logger.log["epoch"][-history:]))
    losses = np.array(logger.log["reconLoss"][-history:])
    iters = np.array(logger.log["iter"][-history:])
    uepochs = np.unique(epochs)

    epoch_losses = np.zeros(len(uepochs))
    epoch_iters = np.zeros(len(uepochs))
    i = 0
    for uepoch in uepochs:
        inds = np.equal(epochs, uepoch)
        loss = np.mean(losses[inds])
        epoch_losses[i] = loss
        epoch_iters[i] = np.mean(iters[inds])
        i += 1

    mval = np.mean(losses)

    plt.figure(figsize=(figx, figy), dpi=dpi)
    plt.plot(x, y, label="reconLoss")
    plt.plot(epoch_iters, epoch_losses, color="darkorange", label="epoch avg")
    plt.plot(
        [np.min(iters), np.max(iters)],
        [mval, mval],
        color="darkorange",
        linestyle=":",
        label="window avg",
    )

    plt.legend()
    plt.title("Short history")
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def embeddings(embedding, save_path):
    plt.figure(figsize=(figx, figy), dpi=dpi)
    colors = plt.get_cmap("plasma")(np.linspace(0, 1, embedding.shape[0]))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=2, color=colors)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.axis("equal")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("latent space embedding")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def embedding_variation(embedding_paths, figsize=(8, 4), save_path=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = cm.viridis(np.linspace(0, 1, len(embedding_paths)))

    for path, color in zip(embedding_paths, colors):
        embeddings = pickle.load(open(path, "rb"))

        var_dims = np.sort(np.var(embeddings, axis=0))[::-1]
        ax1.plot(var_dims, color=color)
        ax1.set_xlabel("dimension #")
        ax1.set_ylabel("dimension variation")
        ax1.set_ylim(0, 1.05)

        ax2.plot(np.cumsum(var_dims) / np.sum(var_dims), color=color)
        ax2.set_xlabel("dimension #")
        ax2.set_ylabel("cumulative variation")

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()


def qq_plot(stats_x, stats_y):
    u_vals = np.unique(np.hstack([stats_x, stats_y]))

    x_vals = list()
    y_vals = list()
    for x in u_vals:
        x_vals.append(stats.percentileofscore(stats_x, x))
        y_vals.append(stats.percentileofscore(stats_y, x))

    plt.plot(x_vals, y_vals, color="r")
    plt.plot([0, 100], [0, 100], "--", color="k")


def scatter_hist(
    stats_list_x, stats_list_y, labels=None, nbins=200, s=5, prct_bounds=[0.1, 99.9]
):

    if labels is None:
        labels = [None] * len(stats_list_x)

    x_lb = np.percentile(np.hstack(stats_list_x), prct_bounds[0])
    x_hb = np.percentile(np.hstack(stats_list_x), prct_bounds[1])
    x_bins = np.linspace(x_lb, x_hb, nbins)

    y_lb = np.percentile(np.hstack(stats_list_y), prct_bounds[0])
    y_hb = np.percentile(np.hstack(stats_list_y), prct_bounds[1])
    y_bins = np.linspace(y_lb, y_hb, nbins)

    ###
    # Parts C&P'd from https://matplotlib.org/examples/pylab_examples/scatter_hist.html
    ###
    nullfmt = NullFormatter()  # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axScatter = plt.axes(rect_scatter)

    # the scatter plot:
    alpha = 1 / len(stats_list_x)
    colors = ["c", "m", "c", "y"]
    for x, y, color, label in zip(stats_list_x, stats_list_y, colors, labels):

        axHistx.hist(x, bins=x_bins, color=color, alpha=alpha)
        axHisty.hist(y, bins=y_bins, color=color, alpha=alpha, orientation="horizontal")

        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())

        axScatter.scatter(x, y, s=s, c=color, alpha=alpha, label=label)

        axScatter.set_xlim((x_lb, x_hb))
        axScatter.set_ylim((y_lb, y_hb))

    if not np.all(np.array([label is None for label in labels])):
        plt.legend(loc="upper right")


def plot_dim_variation(X):
    stds = np.std(X, axis=0)
    sorted_inds = np.argsort(stds)[::-1]

    plt.plot(stds[sorted_inds])

    return stds, sorted_inds
