import torch

"""
Implements Improved Precision and Recall Metric for Assessing Generative Models
https://arxiv.org/abs/1904.06991
"""


def pairwise_distances(x, y=None):
    """
    from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    TODO: switch to using torch.pdist and torch.cdist
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def manifold_estimate(phi_a, phi_b, k=3):
    """phi_a and phi_b are N_a x D and N_b x D sets of features, and
    k is the nearest neighbor setting for manifold esitmation."""
    daa = pairwise_distances(phi_a, phi_a)
    dab = pairwise_distances(phi_a, phi_b)
    r_a, _ = torch.kthvalue(daa, k + 1, dim=0)
    out, _ = torch.max(dab < r_a.view(-1, 1), dim=0)
    return out.float().mean().item()


def precision_recall(phi_r, phi_g, k=3):
    """phi_r and phi_s are N_a x D and N_b x D sets of features
    from real and generated images respectively."""
    return {
        "precision": manifold_estimate(phi_r, phi_g, k=k),
        "recall": manifold_estimate(phi_g, phi_r, k=k),
    }
