import torch.nn as nn

from . import utils

import torch
import math


class KLDLoss(nn.Module):
    # computes kl loss against a reference isotropic gaussian distribution
    def __init__(self, reduction):
        super(KLDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mu, logvar):

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if self.reduction == "sum":
            pass
        elif self.reduction == "batch":  # sum then divide over batch size
            kld = kld / mu.shape[0]
        elif self.reduction == "mean":  # divide over all elements
            kld = kld / torch.numel(mu)

        return kld


class BatchLoss(nn.Module):
    # makes a loss function, and divides by the number of elements in the batch
    def __init__(self, loss, loss_kwargs):
        super(BatchLoss, self).__init__()

        self.loss = utils.load_object(loss, loss_kwargs)

    def forward(self, *args):
        return self.loss(*args) / args[0].shape[0]


class ListMSELoss(nn.Module):
    # computes mse loss over all items
    def __init__(self, **kwargs):
        super(ListMSELoss, self).__init__()

        self.mseloss = nn.MSELoss(**kwargs)

    def forward(self, input_list, target_list):

        losses = torch.zeros(1).type_as(target_list[0])

        for i, t in zip(input_list, target_list):
            losses = losses + self.mseloss(i, t)

        return losses


class BatchMSELoss(nn.Module):
    # computes mse loss and averages over batch size
    def __init__(self):
        super(BatchMSELoss, self).__init__()

        self.mseloss = nn.MSELoss(reduction="sum")

    def forward(self, input, target):
        losses = self.mseloss(input, target)
        losses = losses / input.shape[0]

        return losses


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

        self.mseloss = nn.MSELoss(**kwargs)

    def forward(self, input, target):
        target = target.float()

        return self.mseloss(input, target)


class NormalizedMSELoss(nn.MSELoss):
    def __init__(self, **kwargs):
        super(NormalizedMSELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        target = target.float()

        # dat = torch.cat([input, target], 0)

        mu = torch.mean(target.view(target.shape[0], -1), 1)
        sigma = torch.std(target.view(target.shape[0], -1), 1)

        for i in range(len(target.shape) - 1):
            mu = torch.unsqueeze(mu, -1)
            sigma = torch.unsqueeze(sigma, -1)

        # import pdb; pdb.set_trace()

        input = (input - mu) / sigma
        target = (target - mu) / sigma

        return super().forward(input, target)


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
    # mask = x == 0
    # x.masked_scatter_(mask, x[mask] + eps)

    # mask = x == 1
    # x.masked_scatter_(mask, x[mask] - eps)

    # logC = torch.log(2*torch.atanh(1-2*x)/(1-2*x))
    # but there's no torch.atanh so we use the alternate form
    # of arctanh(z) = 1/2 log((1+z)/(1-z))
    # ==> arctanh(1-2x) = 1/2 log((1+(1-2x))/(1-(1-2x)))
    #                   = 1/2 log((2-2x)/(2x))
    #                   = 1/2 log((1-x)/x)
    # ==> logC = torch.log(2*torch.atanh(1-2*x)/(1-2*x))
    #          = torch.log(torch.log((1-x)/x)/(1-2*x))
    #          = torch.log(torch.log(x/(1-x))/(2*x-1))

    # logC = torch.log((torch.log(x / (1.0 - x))) / (2.0 * x - 1.0))

    x_2 = x.clone().detach()

    logC = torch.log((torch.log((x + eps) / (1.0 - (x + eps)))) / (2.0 * x - 1.0))

    if not torch.all(x == x_2):
        import pdb

        pdb.set_trace()

    # taylor expand around x = 0.5 because of the numerical instability
    # terms to fourth order are accurate to float precision on the interval [0.45,0.55]
    def taylor(y, y_0):
        c_0 = torch.log(torch.tensor(2.0).type_as(y))
        c_2 = 4.0 / 3.0
        c_4 = 104.0 / 45.0

        diff2 = (y - y_0) ** 2

        return c_0 + c_2 * diff2 + c_4 * diff2 ** 2

    mask = torch.abs(x - taylor_center) < taylor_radius
    # logC.masked_scatter_(mask, taylor(x[mask], taylor_center))
    logC[mask] = taylor(x[mask], taylor_center)

    return logC


class ContinuousBCELoss(nn.Module):
    r"""Creates a criterion that is the Continuous BCELoss of
    The continuous Bernoulli: fixing a pervasive error in variational autoencoders
    https://arxiv.org/abs/1907.06845v1

    """

    def __init__(self, **kwargs):
        super(ContinuousBCELoss, self).__init__()

        self.BCELoss = nn.BCELoss(**kwargs)

    def forward(self, input, target):

        C = logC_taylor(input)

        BCE = self.BCELoss(input, target)

        if self.BCELoss.reduction == "mean":
            C = torch.mean(C)
        elif self.BCELoss.reduction == "sum":
            C = torch.sum(C)
        elif self.BCELoss.reduction == "none":
            pass

        return BCE - C


class GaussianNLL(nn.Module):
    """Loss function to capture heteroscedastic aleatoric uncertainty."""

    def __init__(self, reduction="mean", eps=1e-8):
        super(GaussianNLL, self).__init__()

        self.reduction = reduction
        self.eps = eps

        self.log2pi = torch.log(torch.tensor(2 * math.pi))

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        """Calculates loss.
        Parameters
        ----------
        x_hat
           Batched, n-channel model output (estimate variance from the loss over the batch).
           or
           Batched, n*2-channel model output (assume the 2nd channel is variance).

        x
           Batched, n-channel target output.
        """

        if x.shape[1] == x_hat.shape[1]:
            mu = x_hat
            var = torch.mean((x - mu).pow(2))
        else:
            mu = x_hat[:, 0::2, :, :]
            var = x_hat[:, 1::2, :, :] + self.eps

        self.log2pi = self.log2pi.type_as(x)

        loss = -(
            -0.5 * self.log2pi - 0.5 * torch.log(var) - 0.5 * ((x - mu).pow(2) / var)
        )

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass

        return loss


# class GaussianLikelihoodLoss(nn.Module):
#     """Loss function to capture heteroscedastic aleatoric uncertainty."""

#     def __init__(self, reduction="mean"):
#         self.reduction = reduction
#         super(GaussianLikelihoodLoss, self).__init__()

#     def forward(self, y_hat_batch: torch.Tensor, y: torch.Tensor):
#         """Calculates loss.
#         Parameters
#         ----------
#         y_hat_batch
#            Batched, 2-channel model output.
#         y_batch
#            Batched, 1-channel target output.
#         """
#         mu = y_hat_batch[:, 0::2]
#         log_var = nn.functional.softplus(y_hat_batch[:, 1::2])

#         var = torch.exp(log_var)
#         log_scale = torch.log(torch.sqrt(var))

#         loss = (
#             ((y - mu) ** 2) / (2 * var)
#             - log_scale
#             - torch.log(torch.sqrt(2 * torch.tensor(math.pi)))
#         )

#         if self.reduction == "mean":
#             loss = torch.mean(loss)
#         elif self.reduction == "sum":
#             loss = torch.sum(loss)
#         elif self.reduction == "none":
#             pass

#         return loss


class HeteroscedasticLoss(nn.Module):
    """Loss function to capture heteroscedastic aleatoric uncertainty.
    Like GaussianLikelihoodLoss but with all the constants removed
    https://arxiv.org/pdf/1703.04977.pdf
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(HeteroscedasticLoss, self).__init__()

        # self.eps = eps

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
        log_var = y_hat_batch[:, 1::2, :, :]

        loss = (0.5 * torch.exp(-log_var) * (mu - y_batch).pow(2)) + 0.5 * log_var

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass

        return loss


class CrossEntropyLoss(nn.Module):
    """General cross entropy loss"""

    def __init__(self, reduction="mean", eps=1e-8):
        self.reduction = reduction
        super(CrossEntropyLoss, self).__init__()

        self.eps = eps

    def forward(self, y_hat_batch: torch.Tensor, y: torch.Tensor):
        """
        y_hat_batch and y each sum to 1
        """

        # https://en.wikipedia.org/wiki/Cross_entropy#Relation_to_log-likelihood

        loss = -((y + self.eps) * torch.log(y_hat_batch + self.eps))

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass

        return loss


class ChannelMSELoss(nn.Module):
    """MSE over channels, then reduce some specified way"""

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(ChannelMSELoss, self).__init__()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):

        # https://en.wikipedia.org/wiki/Cross_entropy#Relation_to_log-likelihood

        x = x.view(x.shape[0], x.shape[1], -1)

        x_hat = x_hat.view(x_hat.shape[0], x_hat.shape[1], -1)

        loss = torch.mean((x - x_hat) ** 2, 2)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass

        return loss


class ChannelDirichletNLL(nn.Module):
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(ChannelDirichletNLL, self).__init__()

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        """Calculates loss.
        Parameters
        ----------
        x_hat
           Batched, concentration parameters, must be postive real
        x
           Batched, multinomial, must sum to 1.
        """

        x = x.view(x.shape[0], x.shape[1], -1) + 1e-8
        x_hat = x_hat.view(x_hat.shape[0], x_hat.shape[1], -1)

        loss = []

        for b, b_hat in zip(x, x_hat):

            c_loss = torch.stack(
                [
                    torch.distributions.dirichlet.Dirichlet(c_hat).log_prob(c)
                    for (c, c_hat) in zip(b, b_hat)
                ]
            )

            loss.append(c_loss)

        loss = -torch.stack(loss)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass

        return loss
