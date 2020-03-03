from tqdm import tqdm
import torch

from ..utils import utils, reparameterize

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

        x, classes, ref = sampler(mode, [0])

        x = x.cuda()
        ref = ref.cuda()
        classes_onehot = utils.index_to_onehot(classes, dp.get_n_classes()).cuda()

        with torch.no_grad():
            zAll = enc(x, ref, classes_onehot)

        ndims = torch.prod(torch.tensor(zAll[0].shape[1:]))

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
            ref = ref.cuda()
            classes_onehot = utils.index_to_onehot(classes, dp.get_n_classes()).cuda()

            with torch.no_grad():
                zAll = enc(x, ref, classes_onehot)

            recons_target = get_recons(
                x,
                ref,
                classes_onehot,
                dec,
                zAll,
                n_recon_samples,
                recon_loss=recon_loss,
            )

            klds = torch.stack(
                [kld_loss(mu, sigma) for mu, sigma in zip(zAll[0], zAll[1])]
            )

            embeddings_target_mu.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[0].cpu()
            )

            embeddings_target_sigma.index_copy_(
                0, torch.LongTensor(data_iter[i]), zAll[1].cpu()
            )

            embeddings_classes.index_copy_(0, data_iter[i], classes)

            embeddings_target_kld.index_copy_(
                0, torch.LongTensor(data_iter[i]), klds.cpu()
            )

            embeddings_target_recons.index_copy_(
                0, torch.LongTensor(data_iter[i]), recons_target
            )

        embedding[mode] = {}
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


def get_recons(x, ref, classes_onehot, dec, zAll, n_recon_samples, recon_loss):

    recons = torch.zeros(x.shape[0], n_recon_samples)

    for i in range(n_recon_samples):
        zOut = reparameterize(zAll[0], zAll[1])

        with torch.no_grad():
            xHat = dec(zOut, ref, classes_onehot)

        recons[:, i] = torch.stack(
            [recon_loss(xHat[[ind]], x[[ind]]) for ind in range(len(x))]
        )

    return recons
