import torch
from ..external.pytorch_fid.inception import InceptionV3
from ..external.pytorch_fid import fid_score
import os
import pathlib

"""
Implements Improved Precision and Recall Metric for Assessing Generative Models
https://arxiv.org/abs/1904.06991
"""


def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


def manifold_estimate(phi_a, phi_b, k=[3], batch_size=None):
    """
    phi_a and phi_b are N_a x D and N_b x D sets of features,
    k is the nearest neighbor setting for knn manifold estimation,
    batch_size is the number of pairwise distance computations done at once (None = do them all at once)
    """
    if batch_size is None:
        d_aa = my_cdist(phi_a, phi_a)
        r_phi = {kappa: torch.kthvalue(d_aa, kappa + 1, dim=1)[0] for kappa in k}
        d_ba = my_cdist(phi_b, phi_a)
        b_in_a = {
            kappa: torch.any(d_ba <= rp, dim=1).float() for kappa, rp in r_phi.items()
        }
    else:
        r_phi = {kappa: torch.empty_like(phi_a[:, 0]) for kappa in k}
        for batch_inds in torch.split(torch.arange(len(phi_a)), batch_size):
            d_aa = my_cdist(phi_a[batch_inds], phi_a)
            for kappa in k:
                r_phi[kappa][batch_inds] = torch.kthvalue(d_aa, kappa + 1, dim=1)[0]
        b_in_a = {kappa: torch.empty_like(phi_b[:, 0]) for kappa in k}
        for batch_inds in torch.split(torch.arange(len(phi_b)), batch_size):
            d_ba = my_cdist(phi_b[batch_inds], phi_a)
            for kappa in k:
                b_in_a[kappa][batch_inds] = torch.any(
                    d_ba <= r_phi[kappa], dim=1
                ).float()
    return {key: value.mean().item() for key, value in b_in_a.items()}


def precision_recall(phi_r, phi_g, k=[3], batch_size=None):
    """
    phi_r and phi_s are N_a x D and N_b x D sets of features
    from real and generated images respectively,
    k is a list of integers for the nearest neighbor setting for knn manifold estimation.
    returns a dict {k_1:{'precision':float, 'recall':float}, k_2:{'precision':float, 'recall':float}, ...}
    """
    d = {
        "precision": manifold_estimate(phi_r, phi_g, k=k, batch_size=batch_size),
        "recall": manifold_estimate(phi_g, phi_r, k=k, batch_size=batch_size),
    }

    result = {}
    for k1, subdict in d.items():
        for k2, v in subdict.items():
            result.setdefault(k2, {})[k1] = v

    return result


def calculate_inception_pr_given_paths(
    paths, k=[3], batch_size=1, cuda=True, dims=2048, verbose=False
):
    # the is constructed to imitate `calculate_fid_given_paths` from integrated_cell.external.pytorch_fid.fidscore.calculate_fid_given_paths

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    path = pathlib.Path(paths[0])
    files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    act_real = fid_score.get_activations(files, model, batch_size, dims, cuda, verbose)

    path = pathlib.Path(paths[1])
    files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    act_gen = fid_score.get_activations(files, model, batch_size, dims, cuda, verbose)

    act_real = torch.tensor(act_real)
    act_gen = torch.tensor(act_gen)

    return precision_recall(act_real, act_gen, k, batch_size=None)


def calculate_inception_pr_given_paired_paths(
    paths_real, paths_gen, k=[3], batch_size=1, cuda=True, dims=2048, verbose=False
):
    # the is constructed to imitate `calculate_fid_given_paths` from integrated_cell.external.pytorch_fid.fidscore.calculate_fid_given_paths

    for p in paths_real + paths_gen:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    prs = list()
    for i, [path_real, path_gen] in enumerate(zip(paths_real, paths_gen)):
        path = pathlib.Path(path_real)
        files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
        act_real = fid_score.get_activations(
            files, model, batch_size, dims, cuda, verbose
        )

        path = pathlib.Path(path_gen)
        files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
        act_gen = fid_score.get_activations(
            files, model, batch_size, dims, cuda, verbose
        )

        act_real = torch.tensor(act_real)
        act_gen = torch.tensor(act_gen)

        prs.append(precision_recall(act_real, act_gen, k, batch_size=None))

    return prs
