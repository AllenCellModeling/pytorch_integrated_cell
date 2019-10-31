import numpy as np

# import matplotlib as mpl
from matplotlib.ticker import NullFormatter
from matplotlib import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle

import torch

# mpl.use("Agg")  # noqa

dpi = 100
figx = 6
figy = 4.5


def history(logger, save_path):

    # Figure out the default color order, and use these for the plots
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(figx, figy), dpi=dpi)

    ax = plt.gca()

    plts = list()

    x = logger.log["iter"]
    y = logger.log["reconLoss"]

    keep_inds = ~np.any(np.stack([np.isnan(y), np.isinf(y)]), 0)

    x = np.array(x)[keep_inds]
    y = np.array(y)[keep_inds]

    # Plot reconstruction loss
    plts += ax.plot(x, y, label="reconLoss", color=colors[0])

    plt.ylabel("reconLoss")

    ax_max = np.percentile(y, 99)
    ax_min = np.percentile(y, 0)

    ax.set_ylim([ax_min, ax_max])

    # Plot everything else that isn't below
    do_not_print = ["epoch", "iter", "time", "reconLoss"]

    # Print off the reconLoss on it's own scale
    ax2 = plt.gca().twinx()

    y_vals = list()

    i = 1
    for field in logger.fields:
        if field not in do_not_print:
            x = logger.log["iter"]
            y = logger.log[field]

            keep_inds = ~np.any(np.stack([np.isnan(y), np.isinf(y)]), 0)

            x = np.array(x)[keep_inds]
            y = np.array(y)[keep_inds]

            plts += ax2.plot(x, y, label=field, color=colors[i])
            y_vals.append(y)
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

        axHistx.hist(x, bins=x_bins, color=color, alpha=alpha, density=True)
        axHisty.hist(
            y,
            bins=y_bins,
            color=color,
            alpha=alpha,
            density=True,
            orientation="horizontal",
        )

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


def scatter_im(
    X,
    imfunc,
    zoom=1,
    dims_to_plot=[0, 1],
    ax=None,
    inset=False,
    inset_offset=0.15,
    inset_width_and_height=0.1,
    plot_range=[0.05, 99.95],
    inset_colors=None,
    inset_scatter_size=25,
    inset_title=None,
    inset_clims=None,
):
    # Adapted from https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points/53851017

    if ax is None:
        ax = plt.gca()

    artists = []

    for i in range(X.shape[0]):
        im = OffsetImage(imfunc(i), zoom=zoom)
        ab = AnnotationBbox(
            im,
            (X[i, dims_to_plot[0]], X[i, dims_to_plot[1]]),
            xycoords="data",
            frameon=False,
        )
        artists.append(ax.add_artist(ab))

    ax.update_datalim(X[:, dims_to_plot])
    ax.autoscale()

    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    x_lb, x_hb = np.percentile(X[:, dims_to_plot[0]], plot_range)
    y_lb, y_hb = np.percentile(X[:, dims_to_plot[1]], plot_range)

    x_pad = (x_hb - x_lb) * 0.1
    y_pad = (y_hb - y_lb) * 0.1

    ax.set_xlim(x_lb - x_pad, x_hb + x_pad)
    ax.set_ylim(y_lb - y_pad, y_hb + y_pad)

    ax.set_facecolor((0, 0, 0))
    nullfmt = NullFormatter()
    ax.xaxis.set_major_formatter(nullfmt)
    ax.yaxis.set_major_formatter(nullfmt)

    ax.axis("off")

    ax_inset = None
    if inset:

        offset = 0.15

        inset = [
            offset,
            1 - inset_offset - inset_width_and_height,
            inset_width_and_height,
            inset_width_and_height,
        ]
        ax_inset = plt.axes(inset)

        if inset_clims is None:
            inset_clims = np.percentile(inset_colors, [0, 100])

        ax_inset.scatter(
            X[:, dims_to_plot[0]],
            X[:, dims_to_plot[1]],
            s=inset_scatter_size,
            c=inset_colors,
            vmin=inset_clims[0],
            vmax=inset_clims[1],
        )

        for k in ax_inset.spines:
            ax_inset.spines[k].set_color("w")

        #         ax_inset.set_facecolor('w')
        ax_inset.xaxis.label.set_color("w")
        ax_inset.yaxis.label.set_color("w")

        ax_inset.tick_params(axis="x", colors="w")
        ax_inset.tick_params(axis="y", colors="w")

        if inset_title is not None:
            ax_inset.set_title(inset_title)

        return ax, ax_inset

    return ax


def tensor2im(im, scale_channels=True, scale_global=True, color_transform=None):
    # assume #imgs by #channels by Y by X by Z image
    #
    # scale_channels scales channels across all images to range (0, 1)
    # scale_global scales max intensity across all images to range (0, 1)
    # color_transform is #channels by 3 matrix that corresponds to RGB color for each channel
    # todo: fine tune color choices

    def im_adjust(im, scale_channels):
        # assume 3 by x by y (by z) image

        if len(im.shape) == 4:
            # if 3D, then max project
            im = torch.max(im, 3)[0]

        if scale_channels:
            for i in range(im.shape[0]):
                if torch.sum(im[i]) > 0:
                    im[i] = im[i] / torch.max(im[i])

        return im

    im = torch.cat([im_adjust(i, scale_channels) for i in im], 2)

    im = im.clone().cpu().detach().numpy().transpose([1, 2, 0])

    im_shape = np.array(im.shape)
    n_channels = im_shape[2]

    if color_transform is None:
        if n_channels == 3:
            # do magenta-yellow-cyan instead of RGB
            color_transform = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]).T
        elif n_channels == 1:
            # do white
            color_transform = np.array([[1, 1, 1]])
        else:
            # pick colors from HSV
            color_transform = plt.get_cmap("jet")(np.linspace(0, 1, n_channels))[:, 0:3]

    im_reshape = im.reshape([np.prod(im_shape[0:2]), im_shape[2]]).T

    im_recolored = np.matmul(color_transform.T, im_reshape).T

    im_shape[2] = 3
    im = im_recolored.reshape(im_shape)

    if scale_global:
        im = im / np.max(im)
        # im[im > 1] = 1

    return im


def imshow(im, scale_channels=True, scale_global=True):
    # assume CYX image
    im = tensor2im(im, scale_channels, scale_global)
    plt.imshow(im)
    plt.axis("off")
    plt.show()


# def tensor2img(img):

#     img = img.numpy()
#     im_out = list()
#     for i in range(0, img.shape[0]):
#         im_out.append(img[i])

#     img = np.concatenate(im_out, 2)

#     if len(img.shape) == 3:
#         img = np.expand_dims(img, 3)

#     colormap = "hsv"

#     colors = plt.get_cmap(colormap)(np.linspace(0, 1, img.shape[0] + 1))

#     # img = np.swapaxes(img, 2,3)
#     img = imgtoprojection(np.swapaxes(img, 1, 3), colors=colors, global_adjust=True)
#     img = np.swapaxes(img, 0, 2)

#     return img
