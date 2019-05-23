import torch

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


def manifold_estimate(phi_a, phi_b, k=3, batch_size=None):
    """
    phi_a and phi_b are N_a x D and N_b x D sets of features,
    k is the nearest neighbor setting for knn manifold estimation,
    batch_size is the number of pairwise distance computations done at once (None = do them all at once)
    """
    if batch_size is None:
        r_phi = torch.kthvalue(my_cdist(phi_a, phi_a), k + 1, dim=1)[0]
        d_ba = my_cdist(phi_b, phi_a)
        b_in_a = torch.any(d_ba <= r_phi, dim=1).float()
    else:
        r_phi = torch.empty_like(phi_a[:, 0])
        for batch_inds in torch.split(torch.arange(len(phi_a)), batch_size):
            d_aa = my_cdist(phi_a[batch_inds], phi_a)
            r_phi[batch_inds] = torch.kthvalue(d_aa, k + 1, dim=1)[0]
        b_in_a = torch.empty_like(phi_b[:, 0])
        for batch_inds in torch.split(torch.arange(len(phi_b)), batch_size):
            d_ba = my_cdist(phi_b[batch_inds], phi_a)
            b_in_a[batch_inds] = torch.any(d_ba <= r_phi, dim=1).float()
    return b_in_a.mean().item()


def precision_recall(phi_r, phi_g, k=3, batch_size=None):
    """
    phi_r and phi_s are N_a x D and N_b x D sets of features
    from real and generated images respectively, and k is the nearest
    neighbor setting for knn manifold estimation.
    """
    return {
        "precision": manifold_estimate(phi_r, phi_g, k=k, batch_size=batch_size),
        "recall": manifold_estimate(phi_g, phi_r, k=k, batch_size=batch_size),
    }
