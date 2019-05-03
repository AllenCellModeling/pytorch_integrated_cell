import torch

"""
Implements Improved Precision and Recall Metric for Assessing Generative Models
https://arxiv.org/abs/1904.06991
"""


def manifold_estimate(phi_a, phi_b, k=3):
    """phi_a and phi_b are N_a x D and N_b x D sets of features, and
    k is the nearest neighbor setting for manifold esitmation."""
    daa = torch.cdist(phi_a, phi_a)
    r_a, _ = torch.kthvalue(daa, k + 1, dim=0)
    del daa
    dab = torch.cdist(phi_a, phi_b)
    out, _ = torch.max(dab < r_a.view(-1, 1), dim=0)
    return out.float().mean().item()


def precision_recall(phi_r, phi_g, k=3):
    """phi_r and phi_s are N_a x D and N_b x D sets of features
    from real and generated images respectively."""
    return {
        "precision": manifold_estimate(phi_r, phi_g, k=k),
        "recall": manifold_estimate(phi_g, phi_r, k=k),
    }
