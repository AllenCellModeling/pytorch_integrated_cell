from tqdm import tqdm
import torch

from ..utils import reparameterize

from ..losses import KLDLoss


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

    kld_loss = KLDLoss(reduction="sum")

    if sampler is None:

        def sampler(mode, inds):
            return dp.get_sample(mode, inds)

    enc.eval()
    dec.eval()

    embedding = dict()

    for mode in modes:
        ndat = dp.get_n_dat(mode)

        x = sampler(mode, [0])
        x = x.cuda()

        with torch.no_grad():
            zAll = enc(x)

        ndims = torch.prod(torch.tensor(zAll[0].shape[1:]))

        embeddings_ref_mu = torch.zeros(ndat, ndims)
        embeddings_ref_sigma = torch.zeros(ndat, ndims)

        embeddings_ref_recons = torch.zeros(ndat, n_recon_samples)
        embeddings_ref_kld = torch.zeros(ndat)

        inds = list(range(0, ndat))
        data_iter = [
            torch.LongTensor(inds[i : i + batch_size])  # noqa
            for i in range(0, len(inds), batch_size)  # noqa
        ]

        for i in tqdm(range(0, len(data_iter))):
            batch_size = len(data_iter[i])

            x = sampler(mode, data_iter[i])
            x = x.cuda()

            with torch.no_grad():
                zAll = enc(x)

            recons_ref = get_recons(
                x, dec, zAll, n_recon_samples, recon_loss=recon_loss
            )

            klds = torch.stack(
                [kld_loss(mu, sigma) for mu, sigma in zip(zAll[0], zAll[1])]
            )

            embeddings_ref_mu.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[0].cpu()
            )

            embeddings_ref_sigma.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[1].cpu()
            )

            embeddings_ref_kld.index_copy_(
                0, torch.LongTensor(data_iter[i]), klds.cpu()
            )

            embeddings_ref_recons.index_copy_(
                0, torch.LongTensor(data_iter[i]), recons_ref
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

        for mode in embedding:
            for component in embedding[mode]:
                for thing in embedding[mode][component]:
                    embedding[mode][component][thing] = (
                        embedding[mode][component][thing].cpu().detach()
                    )

    return embedding


def get_recons(x, dec, zAll, n_recon_samples, recon_loss):

    recons = torch.zeros(x.shape[0], n_recon_samples)

    for i in range(n_recon_samples):
        zOut = reparameterize(zAll[0], zAll[1])

        with torch.no_grad():
            xHat = dec(zOut)

        recons[:, i] = torch.stack(
            [recon_loss(xHat[[ind]], x[[ind]]) for ind in range(len(x))]
        )

    return recons
