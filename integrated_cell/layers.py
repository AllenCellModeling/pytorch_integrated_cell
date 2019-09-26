import torch
import torch.nn as nn


class ChannelSoftmax(nn.Module):
    # normalizes each channel to sum to one
    def __init__(self):
        super(ChannelSoftmax, self).__init__()

    def forward(self, x):

        # https://stackoverflow.com/questions/44081007/logsoftmax-stability
        # or even better
        # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        n_dims = len(x.shape) - 2

        b = torch.max(x.view(x.shape[0], x.shape[1], -1), dim=2)[0]

        for i in range(n_dims):
            b = torch.unsqueeze(b, -1)

        x_exp = torch.exp(x - b)

        normalize = x_exp.clone()
        for i in range(n_dims):
            normalize = torch.sum(normalize, -1)

        for i in range(n_dims):
            normalize = torch.unsqueeze(normalize, -1)

        softmaxed = x_exp / normalize

        return softmaxed
