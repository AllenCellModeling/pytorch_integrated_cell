from tqdm import tqdm
import torch

from ..models import bvae
from ..utils import utils


def get_latent_embeddings(
    enc, dec, dp, modes=["test"], batch_size=256, n_loss_samples=10, loss=None
):

    enc.eval()
    dec.eval()

    embedding = dict()

    for mode in modes:
        ndat = dp.get_n_dat(mode)

        x, classes, ref = dp.get_sample(mode, [0])

        x = x.cuda()
        classes_onehot = utils.index_to_onehot(classes, dp.get_n_classes()).cuda()

        with torch.no_grad():
            zAll = enc(x, classes_onehot)

        embeddings_ref_mu = torch.zeros(ndat, zAll[0][0].shape[1])
        embeddings_ref_sigma = torch.zeros(ndat, zAll[0][1].shape[1])

        embeddings_ref_losses = torch.zeros(ndat, n_loss_samples)
        embeddings_ref_kld = torch.zeros(ndat)

        embeddings_struct_mu = torch.zeros(ndat, zAll[1][0].shape[1])
        embeddings_struct_sigma = torch.zeros(ndat, zAll[1][1].shape[1])

        embeddings_struct_losses = torch.zeros(ndat, n_loss_samples)
        embeddings_struct_kld = torch.zeros(ndat)

        embeddings_classes = torch.zeros(ndat).long()

        inds = list(range(0, ndat))
        data_iter = [
            inds[i : i + batch_size] for i in range(0, len(inds), batch_size)  # noqa
        ]

        for i in tqdm(range(0, len(data_iter))):
            x, classes, ref = dp.get_sample(mode, data_iter[i])

            x = x.cuda()
            classes_onehot = utils.index_to_onehot(classes, dp.get_n_classes()).cuda()

            with torch.no_grad():
                zAll = enc(x, classes_onehot)

            embeddings_ref_mu.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[0][0].data[:].cpu()
            )
            embeddings_ref_sigma.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[0][1].data[:].cpu()
            )

            embeddings_struct_mu.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[1][0].data[:].cpu()
            )
            embeddings_struct_sigma.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[1][1].data[:].cpu()
            )

            embeddings_classes.index_copy_(0, torch.LongTensor(data_iter[i]), classes)

            losses_all, losses_ref, losses_struct = get_losses(
                x, classes_onehot, dec, zAll, n_loss_samples, loss=loss
            )

            embeddings_ref_kld.index_copy_(
                0,
                torch.LongTensor(data_iter[i]),
                get_klds(zAll[0][0], zAll[0][1]).data[:].cpu(),
            )
            embeddings_ref_losses.index_copy_(
                0, torch.LongTensor(data_iter[i]), losses_ref
            )

            embeddings_struct_kld.index_copy_(
                0, torch.LongTensor(data_iter[i]), get_klds(zAll[1][0], zAll[1][1])
            )
            embeddings_struct_losses.index_copy_(
                0, torch.LongTensor(data_iter[i]), losses_struct
            )

        embedding[mode] = {}
        embedding[mode]["ref"] = {}
        embedding[mode]["ref"]["mu"] = embeddings_ref_mu
        embedding[mode]["ref"]["sigma"] = embeddings_ref_sigma

        embedding[mode]["ref"]["kld"] = embeddings_ref_kld
        embedding[mode]["ref"]["loss"] = embeddings_ref_losses

        embedding[mode]["struct"] = {}
        embedding[mode]["struct"]["mu"] = embeddings_struct_mu
        embedding[mode]["struct"]["sigma"] = embeddings_struct_sigma
        embedding[mode]["struct"]["class"] = embeddings_classes

        embedding[mode]["struct"]["kld"] = embeddings_struct_kld
        embedding[mode]["struct"]["loss"] = embeddings_struct_losses

    return embedding


def get_klds(mus, sigmas):

    klds = torch.zeros(mus.shape[0])

    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):

        kld, _, _ = bvae.kl_divergence(mu.unsqueeze(0), sigma.unsqueeze(0))

        klds[i] = kld[0]

    return klds


def get_losses(x, classes_onehot, dec, zAll, n_loss_samples, loss):

    if loss is None:
        loss = torch.nn.MSELoss(reduction="none")

    reduction_tmp = loss.reduction
    loss.reduction = "none"

    losses_all = torch.zeros(x.shape[0], n_loss_samples)
    losses_ref = torch.zeros(x.shape[0], n_loss_samples)
    losses_struct = torch.zeros(x.shape[0], n_loss_samples)

    for j in range(n_loss_samples):
        zOut = [bvae.reparameterize(zAll[i][0], zAll[i][1]) for i in range(len(zAll))]

        with torch.no_grad():
            xHat = dec([classes_onehot] + zOut)

        squared_errors_per_px = loss(xHat, x)

        tot_errors_per_ch = squared_errors_per_px

        while len(tot_errors_per_ch.shape) > 2:
            tot_errors_per_ch = torch.sum(tot_errors_per_ch, -1)

        mse_per_ch = tot_errors_per_ch / squared_errors_per_px[0, 0].numel()

        mse_all = torch.mean(mse_per_ch, 1)
        mse_ref = torch.mean(mse_per_ch[:, [0, 2]], 1)
        mse_struct = torch.mean(mse_per_ch[:, [1]], 1)

        losses_all[:, j] = mse_all
        losses_ref[:, j] = mse_ref
        losses_struct[:, j] = mse_struct

    loss.reduction = reduction_tmp

    return losses_all, losses_ref, losses_struct
