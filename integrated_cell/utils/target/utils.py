import torch


def sample2im(x, ref):
    n_channels = x.shape[1] + ref.shape[1]

    im = torch.zeros([x.shape[0], n_channels, x.shape[2], x.shape[3], x.shape[4]])

    im[:, [0, 2]] = ref
    im[:, 1] = x

    return im
