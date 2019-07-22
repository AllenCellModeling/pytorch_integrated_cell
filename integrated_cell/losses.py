import torch.nn as nn

from . import utils

import torch
import math


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


def logC(input):

    eps = torch.tensor(1e-7).float()

    mask = input == 0.5
    input[mask] = input[mask] + eps

    mask = input == 0
    input[mask] = input[mask] + eps

    mask = input == 1
    input[mask] = input[mask] - eps

    C = torch.log(-(torch.log(1 - input) - torch.log(input)) / (2 * input - 1))

    return C


def logC_taylor(x, eps=1e-7, taylor_center=0.5, taylor_radius=0.05):

    eps = torch.tensor(eps).type_as(x)
    taylor_center = torch.tensor(taylor_center).type_as(x)
    taylor_radius = torch.tensor(taylor_radius).type_as(x)

    # singular at zero and one, so regularize
    mask = x == 0
    x[mask] = x[mask] + eps

    mask = x == 1
    x[mask] = x[mask] - eps

    # logC = torch.log(2*torch.atanh(1-2*x)/(1-2*x))
    # but there's no torch.atanh so we use the alternate form
    # of arctanh(z) = 1/2 log((1+z)/(1-z))
    # ==> arctanh(1-2x) = 1/2 log((1+(1-2x))/(1-(1-2x)))
    #                   = 1/2 log((2-2x)/(2x))
    #                   = 1/2 log((1-x)/x)
    # ==> logC = torch.log(2*torch.atanh(1-2*x)/(1-2*x))
    #          = torch.log(torch.log((1-x)/x)/(1-2*x))
    #          = torch.log(torch.log(x/(1-x))/(2*x-1))

    logC = torch.log((torch.log(x / (1.0 - x))) / (2.0 * x - 1.0))

    # taylor expand around x = 0.5 because of the numerical instability
    # terms to fourth order are accurate to float precision on the interval [0.45,0.55]
    def taylor(y, y_0):
        c_0 = torch.log(torch.tensor(2.0).type_as(y))
        c_2 = 4.0 / 3.0
        c_4 = 104.0 / 45.0

        diff2 = (y - y_0) ** 2

        return c_0 + c_2 * diff2 + c_4 * diff2 ** 2

    mask = torch.abs(x - taylor_center) < taylor_radius
    taylor_result = taylor(x[mask], taylor_center)
    logC[mask] = taylor_result

    return logC


class ContinuousBCELoss(nn.BCELoss):
    r"""Creates a criterion that is the Continuous BCELoss of
    The continuous Bernoulli: fixing a pervasive error in variational autoencoders
    https://arxiv.org/abs/1907.06845v1

    """

    def __init__(self, **kwargs):
        super(ContinuousBCELoss, self).__init__(**kwargs)

    def forward(self, input, target):

        BCE = super(ContinuousBCELoss, self).forward(input, target)

        C = logC_taylor(input)

        if self.reduction == "mean":
            C = torch.mean(C)
        elif self.reduction == "sum":
            C = torch.sum(C)
        elif self.reduction == "none":
            pass

        return BCE - C


class GaussianLikelihoodLoss(torch.nn.Module):
    """Loss function to capture heteroscedastic aleatoric uncertainty."""

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(GaussianLikelihoodLoss, self).__init__()

    def forward(self, y_hat_batch: torch.Tensor, y: torch.Tensor):
        """Calculates loss.
        Parameters
        ----------
        y_hat_batch
           Batched, 2-channel model output.
        y_batch
           Batched, 1-channel target output.
        """
        mu = y_hat_batch[:, 0::2, :, :]
        log_var = y_hat_batch[:, 1::2, :, :]

        var = torch.exp(log_var)
        log_scale = torch.log(torch.sqrt(var))

        loss = (
            ((y - mu) ** 2) / (2 * var)
            - log_scale
            - torch.log(torch.sqrt(2 * torch.tensor(math.pi)))
        )

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass

        return loss


class HeteroscedasticLoss(torch.nn.Module):
    """Loss function to capture heteroscedastic aleatoric uncertainty.
    Like GaussianLikelihoodLoss but with all the constants removed
    https://arxiv.org/pdf/1703.04977.pdf
    """

    def __init__(self, reduction="mean", eps=1e-5):
        self.reduction = reduction
        super(HeteroscedasticLoss, self).__init__()

        self.eps = eps

    def forward(self, y_hat_batch: torch.Tensor, y_batch: torch.Tensor):
        """Calculates loss.
        Parameters
        ----------
        y_hat_batch
           Batched, 2-channel model output.
        y_batch
           Batched, 1-channel target output.
        """
        mu = y_hat_batch[:, 0::2, :, :]
        log_var = y_hat_batch[:, 1::2, :, :] + self.eps

        loss = (0.5 * (mu - y_batch).pow(2)) / torch.exp(log_var) + 0.5 * log_var

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass

        return loss
