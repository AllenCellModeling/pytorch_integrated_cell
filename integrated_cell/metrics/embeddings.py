from tqdm import tqdm
import torch

from ..models import bvae
from ..utils import utils


def get_latent_embeddings(
    enc,
    dec,
    dp,
    recon_loss,
    modes=["test"],
    batch_size=256,
    n_recon_samples=10,
    sampler=None,
    beta=1,
    channels_ref=[0, 2],
    channels_target=[1],
):

    if sampler is None:

        def sampler(mode, inds):
            return dp.get_sample(mode, inds)

    enc.eval()
    dec.eval()

    embedding = dict()

    for mode in modes:
        ndat = dp.get_n_dat(mode)

        x, classes, ref = sampler(mode, [0])

        x = x.cuda()
        classes_onehot = utils.index_to_onehot(classes, dp.get_n_classes()).cuda()

        with torch.no_grad():
            zAll = enc(x, classes_onehot)

        ndims = torch.prod(torch.tensor(zAll[0][0].shape[1:]))

        embeddings_ref_mu = torch.zeros(ndat, ndims)
        embeddings_ref_sigma = torch.zeros(ndat, ndims)

        embeddings_ref_recons = torch.zeros(ndat, n_recon_samples)
        embeddings_ref_kld = torch.zeros(ndat)

        embeddings_target_mu = torch.zeros(ndat, ndims)
        embeddings_target_sigma = torch.zeros(ndat, ndims)

        embeddings_target_recons = torch.zeros(ndat, n_recon_samples)
        embeddings_target_kld = torch.zeros(ndat)

        embeddings_classes = torch.zeros(ndat).long()

        inds = list(range(0, ndat))
        data_iter = [
            torch.LongTensor(inds[i : i + batch_size])  # noqa
            for i in range(0, len(inds), batch_size)  # noqa
        ]

        for i in tqdm(range(0, len(data_iter))):
            batch_size = len(data_iter[i])

            x, classes, ref = sampler(mode, data_iter[i])

            x = x.cuda()
            classes_onehot = utils.index_to_onehot(classes, dp.get_n_classes()).cuda()

            with torch.no_grad():
                zAll = enc(x, classes_onehot)

            embeddings_ref_mu.index_copy_(
                0, data_iter[i], zAll[0][0].cpu().view([batch_size, -1])
            )
            embeddings_ref_sigma.index_copy_(
                0, data_iter[i], zAll[0][1].cpu().view([batch_size, -1])
            )

            embeddings_target_mu.index_copy_(
                0, data_iter[i], zAll[1][0].cpu().view([batch_size, -1])
            )
            embeddings_target_sigma.index_copy_(
                0, data_iter[i], zAll[1][1].cpu().view([batch_size, -1])
            )

            embeddings_classes.index_copy_(0, data_iter[i], classes)

            recons_ref, recons_target = get_recons(
                x,
                classes_onehot,
                dec,
                zAll,
                n_recon_samples,
                recon_loss=recon_loss,
                channels_ref=channels_ref,
                channels_target=channels_target,
            )

            embeddings_ref_kld.index_copy_(
                0,
                torch.LongTensor(data_iter[i]),
                get_klds(zAll[0][0], zAll[0][1]).data[:].cpu(),
            )
            embeddings_ref_recons.index_copy_(
                0, torch.LongTensor(data_iter[i]), recons_ref
            )

            embeddings_target_kld.index_copy_(
                0, torch.LongTensor(data_iter[i]), get_klds(zAll[1][0], zAll[1][1])
            )
            embeddings_target_recons.index_copy_(
                0, torch.LongTensor(data_iter[i]), recons_target
            )

        embedding[mode] = {}
        embedding[mode]["ref"] = {}
        embedding[mode]["ref"]["mu"] = embeddings_ref_mu
        embedding[mode]["ref"]["sigma"] = embeddings_ref_sigma

        embedding[mode]["ref"]["kld"] = embeddings_ref_kld
        embedding[mode]["ref"]["recon"] = embeddings_ref_recons
        embedding[mode]["ref"]["elbo"] = -(
            torch.mean(embeddings_ref_recons, 1) + beta * embeddings_ref_kld
        )

        embedding[mode]["target"] = {}
        embedding[mode]["target"]["mu"] = embeddings_target_mu
        embedding[mode]["target"]["sigma"] = embeddings_target_sigma
        embedding[mode]["target"]["class"] = embeddings_classes

        embedding[mode]["target"]["kld"] = embeddings_target_kld
        embedding[mode]["target"]["recon"] = embeddings_target_recons

        embedding[mode]["target"]["elbo"] = -(
            torch.mean(embeddings_target_recons, 1) + beta * embeddings_target_kld
        )

        for mode in embedding:
            for component in embedding[mode]:
                for thing in embedding[mode][component]:
                    embedding[mode][component][thing] = (
                        embedding[mode][component][thing].cpu().detach()
                    )

    return embedding


def get_klds(mus, sigmas):

    klds = torch.zeros(mus.shape[0])

    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):

        kld, _, _ = bvae.kl_divergence(mu.unsqueeze(0), sigma.unsqueeze(0))

        klds[i] = kld[0]

    return klds


def get_recons(
    x,
    classes_onehot,
    dec,
    zAll,
    n_recon_samples,
    recon_loss,
    channels_ref=[0, 2],
    channels_target=[1],
):

    recons_ref = torch.zeros(x.shape[0], n_recon_samples)
    recons_target = torch.zeros(x.shape[0], n_recon_samples)

    for i in range(n_recon_samples):
        zOut = [bvae.reparameterize(z[0], z[1]) for z in zAll]

        with torch.no_grad():
            xHat = dec([classes_onehot] + zOut)

        recons_ref[:, i] = torch.stack(
            [
                recon_loss(xHat[[ind], [channels_ref]], x[[ind], [channels_ref]])
                for ind in range(len(x))
            ]
        )

        recons_target[:, i] = torch.stack(
            [
                recon_loss(xHat[[ind], [channels_target]], x[[ind], [channels_target]])
                for ind in range(len(x))
            ]
        )

    return recons_ref, recons_target
