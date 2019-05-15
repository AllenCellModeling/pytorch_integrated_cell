import torch

"""
Implements Improved Precision and Recall Metric for Assessing Generative Models
https://arxiv.org/abs/1904.06991
"""


def manifold_estimate(phi_a, phi_b, k=3, parallel=True):
    """
    phi_a and phi_b are N_a x D and N_b x D sets of features, and
    k is the nearest neighbor setting for knn manifold estimation.
    """
    if parallel:
        r_phi = torch.kthvalue(torch.cdist(phi_a, phi_a), k + 1, dim=0).values
        d_ba = torch.cdist(phi_b, phi_a)
        out = torch.any(d_ba <= r_phi, dim=1)
    else:
        r_phi = torch.empty_like(phi_a[:, 0])
        for i, pa in enumerate(phi_a):
            d_aa = torch.cdist(pa.unsqueeze(dim=0), phi_a)
            r_phi[i] = torch.kthvalue(d_aa, k + 1, dim=1).values

        out = torch.empty_like(phi_b[:, 0])
        for j, pb in enumerate(phi_b):
            d_ab = torch.cdist(pb.unsqueeze(dim=0), phi_a)
            out[j] = torch.any(d_ab <= r_phi)
    return out.float().mean().item()


def precision_recall(phi_r, phi_g, k=3, parallel=True):
    """
    phi_r and phi_s are N_a x D and N_b x D sets of features
    from real and generated images respectively, and k is the nearest
    neighbor setting for knn manifold estimation.
    """
    return {
        "precision": manifold_estimate(phi_r, phi_g, k=k, parallel=parallel),
        "recall": manifold_estimate(phi_g, phi_r, k=k, parallel=parallel),
    }
