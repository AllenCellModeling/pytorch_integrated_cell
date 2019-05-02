import torch.nn as nn

from . import utils

import torch


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

        self.mseloss = nn.MSELoss(**kwargs)

    def forward(self, input, target):
        target = target.float()

        return self.mseloss(input, target)


class ClassMSELoss(nn.Module):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`.

    Same as nn.MSELoss but the 'target' is an integer that gets converted to one-hot.
    """

    def __init__(self, **kwargs):
        super(ClassMSELoss, self).__init__()

        self.mseloss = nn.MSELoss(**kwargs)

    def forward(self, input, target):

        if input.shape[1] > 1:
            target_onehot = utils.index_to_onehot(target, input.shape[1]).float()
        else:
            target_onehot = target.float()

        return self.mseloss(input, target_onehot)


class ListClassMSELoss(nn.Module):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x_i` and target `y` for every element of the list x

    Same as nn.MSELoss but the 'target' is an integer that gets converted to one-hot.
    """

    def __init__(self, **kwargs):
        super(ListClassMSELoss, self).__init__()

        self.class_mseloss = ClassMSELoss(**kwargs)

    def forward(self, input, target):

        loss = torch.zeros(1).type_as(target).float()

        for i in input:
            loss += self.class_mseloss(i, target)

        return loss


class ClassMSELossV2(nn.Module):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`.

    Same as nn.MSELoss but the 'target' is an integer that gets converted to one-hot.
    """

    def __init__(self, **kwargs):
        super(ClassMSELossV2, self).__init__()

    def forward(self, input, target):

        if input.shape[1] > 1:
            target_onehot = utils.index_to_onehot(target, input.shape[1]).float()
        else:
            target_onehot = target.float()

        return torch.sum(torch.sum((input - target_onehot).pow(2), 1).pow(0.5), 0)
