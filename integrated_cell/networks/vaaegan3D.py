from torch import nn
import torch
from ..utils import spectral_norm
from ..models import bvae
from .. import utils

ksize = 4
dstep = 2


def get_activation(activation):
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "prelu":
        return nn.PReLU()


class Enc(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        n_classes,
        n_ref,
        n_channels,
        gpu_ids,
        activation="ReLU",
        pretrained_path=None,
    ):
        super(Enc, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.n_latent_dim = n_latent_dim
        self.n_classes = n_classes
        self.n_ref = n_ref

        if pretrained_path is not None:
            pretrained_net, _, _ = utils.load_network_from_args_path(pretrained_path)
            self.main = pretrained_net.main

            if self.main[0].in_channels != n_channels:
                self.main[0] = nn.Conv3d(n_channels, 64, ksize, dstep, 1)

        else:
            self.main = nn.Sequential(
                nn.Conv3d(n_channels, 64, ksize, dstep, 1),
                nn.BatchNorm3d(64),
                get_activation(activation),
                nn.Conv3d(64, 128, ksize, dstep, 1),
                nn.BatchNorm3d(128),
                get_activation(activation),
                nn.Conv3d(128, 256, ksize, dstep, 1),
                nn.BatchNorm3d(256),
                get_activation(activation),
                nn.Conv3d(256, 512, ksize, dstep, 1),
                nn.BatchNorm3d(512),
                get_activation(activation),
                nn.Conv3d(512, 1024, ksize, dstep, 1),
                nn.BatchNorm3d(1024),
                get_activation(activation),
                nn.Conv3d(1024, 1024, ksize, dstep, 1),
                nn.BatchNorm3d(1024),
                get_activation(activation),
            )

        if self.n_classes > 0:
            self.classOut = nn.Sequential(
                nn.Linear(1024 * int(self.fcsize * 1 * 1), self.n_classes),
                # nn.BatchNorm1d(self.n_classes),
                nn.LogSoftmax(),
            )

        if self.n_ref > 0:
            self.refOutMu = nn.Sequential(
                nn.Linear(1024 * int(self.fcsize * 1 * 1), self.n_ref)
            )

            # self.refOutSigma = nn.Sequential(
            #     nn.Linear(1024*int(self.fcsize*1*1), self.n_ref))

        if self.n_latent_dim > 0:
            self.latentOutMu = nn.Sequential(
                nn.Linear(1024 * int(self.fcsize * 1 * 1), self.n_latent_dim)
            )

            self.latentOutLogSigma = nn.Sequential(
                nn.Linear(1024 * int(self.fcsize * 1 * 1), self.n_latent_dim)
            )

    def forward(self, x, reparameterize=False):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 1024 * int(self.fcsize * 1 * 1))

        xOut = list()

        if self.n_classes > 0:
            xClasses = nn.parallel.data_parallel(self.classOut, x, gpu_ids)
            xOut.append(xClasses)

        if self.n_ref > 0:
            xRefMu = nn.parallel.data_parallel(self.refOutMu, x, gpu_ids)
            xOut.append(xRefMu)

        if self.n_latent_dim > 0:
            xLatentMu = nn.parallel.data_parallel(self.latentOutMu, x, gpu_ids)
            xLatentLogSigma = nn.parallel.data_parallel(
                self.latentOutLogSigma, x, gpu_ids
            )

            if self.training:
                xOut.append([xLatentMu, xLatentLogSigma])
            else:
                xOut.append(
                    bvae.reparameterize(xLatentMu, xLatentLogSigma, add_noise=False)
                )

        return xOut


class Dec(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        n_classes,
        n_ref,
        n_channels,
        gpu_ids,
        activation="ReLU",
        fcsize=2,
        output_padding=(0, 1, 0),
        pretrained_path=None,
    ):
        super(Dec, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = fcsize

        self.n_latent_dim = n_latent_dim
        self.n_classes = n_classes
        self.n_ref = n_ref

        self.output_padding = output_padding

        if pretrained_path is not None:
            pretrained_net, _, _ = utils.load_network_from_args_path(pretrained_path)

            self.fc = pretrained_net.fc
            self.main = pretrained_net.main

            if (
                self.n_latent_dim != pretrained_net.n_latent_dim
                or self.n_classes != pretrained_net.n_classes
                or self.n_ref != pretrained_net.n_ref
            ):

                self.fc = nn.Linear(
                    self.n_latent_dim + self.n_classes + self.n_ref,
                    1024 * int(self.fcsize * 1 * 1),
                )

            if self.main[-2].in_channels != n_channels:
                self.main[-2] = nn.ConvTranspose3d(64, n_channels, ksize, dstep, 1)

        else:

            self.fc = nn.Linear(
                self.n_latent_dim + self.n_classes + self.n_ref,
                1024 * int(self.fcsize * 1 * 1),
            )

            self.main = nn.Sequential(
                nn.BatchNorm3d(1024),
                get_activation(activation),
                nn.ConvTranspose3d(
                    1024, 1024, ksize, dstep, 1, output_padding=output_padding
                ),
                nn.BatchNorm3d(1024),
                get_activation(activation),
                nn.ConvTranspose3d(1024, 512, ksize, dstep, 1),
                nn.BatchNorm3d(512),
                get_activation(activation),
                nn.ConvTranspose3d(512, 256, ksize, dstep, 1),
                nn.BatchNorm3d(256),
                get_activation(activation),
                nn.ConvTranspose3d(256, 128, ksize, dstep, 1),
                nn.BatchNorm3d(128),
                get_activation(activation),
                nn.ConvTranspose3d(128, 64, ksize, dstep, 1),
                nn.BatchNorm3d(64),
                get_activation(activation),
                nn.ConvTranspose3d(64, n_channels, ksize, dstep, 1),
                # nn.BatchNorm3d(n_channels),
                nn.Sigmoid(),
            )

    def forward(self, xIn):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        x = torch.cat(xIn, 1)

        x = self.fc(x)
        x = x.view(x.size()[0], 1024, self.fcsize, 1, 1)

        x = nn.parallel.data_parallel(self.main, x, gpu_ids)

        return x


class DecD(nn.Module):
    def __init__(self, n_classes, n_channels, gpu_ids, **kwargs):
        super(DecD, self).__init__()

        self.noise_std = 0
        for key in "noise_std":
            if key in kwargs:
                setattr(self, key, kwargs[key])

        self.noise = torch.zeros(0)

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.noise = torch.zeros(0)

        self.main = nn.Sequential(
            spectral_norm(nn.Conv3d(n_channels, 64, ksize, dstep, 1)),
            # nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(64, 128, ksize, dstep, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(128, 256, ksize, dstep, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(256, 512, ksize, dstep, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(512, 1024, ksize, dstep, 1)),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(1024, 1024, ksize, dstep, 1)),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = spectral_norm(nn.Linear(1024 * int(self.fcsize), n_classes))

        # if n_classes == 1:
        #     self.nlEnd = nn.Sigmoid()
        # else:
        #     self.nlEnd = nn.LogSoftmax()

    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        if self.noise_std > 0:
            # allocate an appropriately sized variable if it does not exist
            if self.noise.size() != x.size():
                self.noise = torch.zeros(x.size()).type_as(x)

            # sample random noise
            self.noise.normal_(mean=0, std=self.noise_std)
            noise = torch.autograd.Variable(self.noise)

            # add to input
            x = x + noise

        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 1024 * int(self.fcsize))
        x = self.fc(x)

        return x
