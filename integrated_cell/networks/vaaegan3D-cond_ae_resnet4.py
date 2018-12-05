from torch import nn
import torch
from ..utils import spectral_norm
from ..models import bvae
from .. import utils

from ..utils import get_activation
import numpy as np

# conv3x3 ch_in, ch_in/2
# bn
# act
# conv3x3 ch_in/2 ch_out
# bn
# upscale/downscale


class BasicLayer(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=4, dstep=2, padding=1, activation="relu"):
        super(BasicLayer, self).__init__()

        self.main = nn.Sequential(
            spectral_norm(
                nn.Conv3d(ch_in, ch_out, ksize, dstep, padding=padding, bias=False)
            ),
            nn.BatchNorm3d(ch_out),
            get_activation(activation),
        )

    def forward(self, x):
        return self.main(x)


class PadLayer(nn.Module):
    def __init__(self, pad_dims):
        super(PadLayer, self).__init__()

        self.pad_dims = pad_dims

    def forward(self, x):
        if np.sum(self.pad_dims) == 0:
            return x
        else:
            return nn.functional.pad(
                x,
                [0, self.pad_dims[2], 0, self.pad_dims[1], 0, self.pad_dims[0]],
                "constant",
                0,
            )


class DownLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        ksize=4,
        dstep=2,
        activation="relu",
        ch_proj_in=0,
        ch_proj_class=0,
        activation_last=None,
    ):
        super(DownLayer, self).__init__()

        if activation_last is None:
            activation_last = activation

        self.bypass = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=0),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=False)),
        )

        self.resid = nn.Sequential(
            spectral_norm(nn.Conv3d(ch_in, ch_in, 3, 1, padding=1, bias=False)),
            nn.BatchNorm3d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=False)),
            nn.BatchNorm3d(ch_out),
            nn.AvgPool3d(2, stride=2, padding=0),
        )

        self.proj = None
        if ch_proj_in > 0:
            self.proj = BasicLayer(ch_proj_in, ch_out, 1, 1, 0, activation=None)

        self.proj_class = None
        if ch_proj_class > 0:
            self.proj_class = BasicLayer(
                ch_proj_class, ch_out, 1, 1, 0, activation=None
            )

        self.activation = get_activation(activation_last)

    def forward(self, x, x_proj=None, x_class=None):
        x = self.bypass(x) + self.resid(x)

        if self.proj is not None:
            x = x + self.proj(x_proj)

        if self.proj_class is not None:
            x = x + self.proj_class(
                x_class.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            ).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        x = self.activation(x)

        return x


class UpLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        ksize=4,
        dstep=2,
        activation="relu",
        output_padding=0,
        ch_proj_in=0,
        ch_proj_class=0,
        activation_last=None,
    ):
        super(UpLayer, self).__init__()

        if activation_last is None:
            activation_last = activation

        self.bypass = nn.Sequential(
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=False)),
            nn.Upsample(scale_factor=2),
            PadLayer(output_padding),
        )

        self.resid = nn.Sequential(
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=False)),
            nn.BatchNorm3d(ch_out),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_out, ch_out, 3, 1, padding=1, bias=False)),
            PadLayer(output_padding),
            nn.BatchNorm3d(ch_out),
        )

        self.proj = None
        if ch_proj_in > 0:
            self.proj = BasicLayer(ch_proj_in, ch_out, 1, 1, 0, activation=None)

        self.proj_class = None
        if ch_proj_class > 0:
            self.proj_class = BasicLayer(
                ch_proj_class, ch_out, 1, 1, 0, activation=None
            )

        self.activation = get_activation(activation_last)

    def forward(self, x, x_proj=None, x_class=None):
        x = self.bypass(x) + self.resid(x)

        if self.proj is not None:
            x = x + self.proj(x_proj)

        if self.proj_class is not None:
            x = x + self.proj_class(
                x_class.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            ).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        x = self.activation(x)

        return x


class Enc(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        n_classes,
        n_ref,
        n_channels,
        n_channels_target,
        gpu_ids,
        ch_ref=[0, 2],
        ch_target=[1],
    ):
        super(Enc, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.ch_ref = ch_ref
        self.ch_target = ch_target

        self.n_ref = n_ref
        self.n_latent_dim = n_latent_dim

        self.ref_path = nn.ModuleList([DownLayer(n_channels, 64)])

        self.target_path = nn.ModuleList(
            [DownLayer(n_channels_target, 64, ch_proj_in=64, ch_proj_class=n_classes)]
        )

        ch_in = 64
        ch_max = 1024

        for i in range(5):
            ch_out = ch_in * 2
            if ch_out > ch_max:
                ch_out = ch_max

            self.ref_path.append(DownLayer(ch_in, ch_out))
            self.target_path.append(
                DownLayer(ch_in, ch_out, ch_proj_in=ch_out, ch_proj_class=n_classes)
            )

            ch_in = ch_out

        if self.n_ref > 0:
            self.ref_out_mu = nn.Sequential(
                nn.Linear(ch_in * int(self.fcsize * 1 * 1), self.n_ref, bias=False)
            )

            self.ref_out_sigma = nn.Sequential(
                nn.Linear(ch_in * int(self.fcsize * 1 * 1), self.n_ref, bias=False)
            )

        if self.n_latent_dim > 0:
            self.latent_out_mu = nn.Sequential(
                nn.Linear(
                    ch_in * int(self.fcsize * 1 * 1), self.n_latent_dim, bias=False
                )
            )

            self.latent_out_sigma = nn.Sequential(
                nn.Linear(
                    ch_in * int(self.fcsize * 1 * 1), self.n_latent_dim, bias=False
                )
            )

    def forward(self, x, x_class):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        x_ref = x[:, self.ch_ref]
        x_target = x[:, self.ch_target]

        for ref_path, target_path in zip(self.ref_path, self.target_path):
            x_ref = ref_path(x_ref)
            x_target = target_path(x_target, x_ref, x_class)

        x_ref = x_ref.view(x_ref.size()[0], 1024 * int(self.fcsize * 1 * 1))
        x_target = x_target.view(x_target.size()[0], 1024 * int(self.fcsize * 1 * 1))

        xOut = list()

        if self.n_ref > 0:
            xRefMu = self.ref_out_mu(x_ref)
            xRefLogSigma = self.ref_out_sigma(x_ref)

            xOut.append([xRefMu, xRefLogSigma])

        if self.n_latent_dim > 0:
            xLatentMu = self.latent_out_mu(x_target)
            xLatentLogSigma = self.latent_out_sigma(x_target)

            xOut.append([xLatentMu, xLatentLogSigma])

        return xOut


class Dec(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        n_classes,
        n_ref,
        n_channels,
        n_channels_target,
        gpu_ids,
        output_padding=(1, 1, 0),
        pretrained_path=None,
        ch_ref=[0, 2],
        ch_target=[1],
    ):

        super(Dec, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.n_latent_dim = n_latent_dim
        self.n_classes = n_classes
        self.n_ref = n_ref

        self.n_channels = n_channels
        self.n_channels_target = n_channels_target

        self.ch_ref = ch_ref
        self.ch_target = ch_target

        self.ref_fc = nn.Linear(self.n_ref, 1024 * int(self.fcsize * 1 * 1), bias=False)
        self.target_fc = nn.Linear(
            self.n_latent_dim, 1024 * int(self.fcsize * 1 * 1), bias=False
        )
        self.ref_bn_relu = nn.Sequential(nn.BatchNorm3d(1024), nn.ReLU(inplace=True))
        self.target_bn_relu = nn.Sequential(nn.BatchNorm3d(1024), nn.ReLU(inplace=True))

        self.ref_path = nn.ModuleList([])
        self.target_path = nn.ModuleList([])

        l_sizes = [1024, 1024, 512, 256, 128, 64]
        for i in range(len(l_sizes) - 1):
            if i == 0:
                padding = output_padding
            else:
                padding = 0

            self.ref_path.append(
                UpLayer(l_sizes[i], l_sizes[i + 1], output_padding=padding)
            )
            self.target_path.append(
                UpLayer(
                    l_sizes[i],
                    l_sizes[i + 1],
                    output_padding=padding,
                    ch_proj_in=l_sizes[i + 1],
                    ch_proj_class=n_classes,
                )
            )

        self.ref_path.append(
            UpLayer(l_sizes[i + 1], n_channels, activation_last="sigmoid")
        )
        self.target_path.append(
            UpLayer(
                l_sizes[i + 1],
                n_channels_target,
                ch_proj_in=n_channels,
                ch_proj_class=n_classes,
                activation_last="sigmoid",
            )
        )

    def forward(self, x_in):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        x_class, x_ref, x_target = x_in

        x_target = self.target_fc(x_target).view(
            x_target.size()[0], 1024, self.fcsize, 1, 1
        )
        x_target = self.target_bn_relu(x_target)

        x_ref = self.ref_fc(x_ref).view(x_ref.size()[0], 1024, self.fcsize, 1, 1)
        x_ref = self.ref_bn_relu(x_ref)

        for ref_path, target_path in zip(self.ref_path, self.target_path):
            x_ref = ref_path(x_ref)
            x_target = target_path(x_target, x_ref, x_class)

        out_shape = list(x_target.shape)
        out_shape[1] = self.n_channels + self.n_channels_target
        out = torch.zeros(out_shape).type_as(x_ref)

        out[:, self.ch_ref] = x_ref
        out[:, self.ch_target] = x_target

        return out


class DecD(nn.Module):
    def __init__(self, n_classes, n_channels, gpu_ids, noise_std, **kwargs):
        super(DecD, self).__init__()

        self.noise_std = noise_std

        self.noise = torch.zeros(0)

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.noise = torch.zeros(0)

        self.path = nn.ModuleList([])

        l_sizes = [n_channels, 64, 128, 256, 512, 1024, 1024]
        for i in range(len(l_sizes) - 1):
            self.path.append(
                DownLayer(l_sizes[i], l_sizes[i + 1], activation="leakyrelu")
            )

        self.fc = spectral_norm(
            nn.Linear(1024 * int(self.fcsize), n_classes, bias=True)
        )

    def forward(self, x_in):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        if self.noise_std > 0:
            # allocate an appropriately sized variable if it does not exist
            if self.noise.size() != x_in.size():
                self.noise = torch.zeros(x_in.size()).type_as(x_in)

            # sample random noise
            self.noise.normal_(mean=0, std=self.noise_std)
            noise = torch.autograd.Variable(self.noise)

            # add to input
            x = x_in + noise
        else:
            x = x_in

        for layer in self.path:
            x = layer(x)

        x = x.view(x.size()[0], x.shape[1] * int(self.fcsize))
        out = self.fc(x)

        return out
