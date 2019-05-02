from torch import nn
import torch
from ..utils import spectral_norm
from ..models import bvae
from .. import utils

from ..utils import get_activation
import numpy as np

from torch.utils.checkpoint import checkpoint, checkpoint_sequential

import torch.nn.functional as F

# conv 4x4 + upscale/downscale
# bn
# activation
# conv 1x1
# bn


chunks = 3


class BasicLayer(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=4, dstep=2, padding=1, activation="relu"):
        super(BasicLayer, self).__init__()

        self.conv = spectral_norm(
            nn.Conv3d(ch_in, ch_out, ksize, dstep, padding=padding, bias=False)
        )
        self.bn = BatchNorm(ch_out)
        self.activation = get_activation(activation)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


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


class BatchNorm(nn.BatchNorm3d):
    """
    In this way, the stat values can be correct now.
    """

    def __init__(self, *args, **kwargs):
        super(BatchNorm, self).__init__(*args, **kwargs)
        self.prev_running_mean = self.running_mean.new(self.running_mean.size())
        self.prev_running_var = self.running_var.new(self.running_var.size())

    def forward(self, input, in_backward=False):
        self._check_input_dim(input)
        if in_backward:
            self.running_mean.copy_(self.prev_running_mean)
            self.running_var.copy_(self.prev_running_var)
        else:
            self.prev_running_mean.copy_(self.running_mean)
            self.prev_running_var.copy_(self.running_var)

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


# class BatchNorm(nn.Module):
#     def __init__(self, ch):
#         super(BatchNorm, self).__init__()

#         self.BatchNorm = nn.BatchNorm3d(ch)

#     def forward(self, x):

#         #this is the dumb hack I need to do to do parallel and save memory
# #         self.BatchNorm.cpu()

# #         cuda_device = x.get_device()

# #         x = x.cpu()

#         x = self.BatchNorm(x)

# #         x = x.cuda(cuda_device)

# #         self.BatchNorm.cuda(0)

#         return x


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
            spectral_norm(nn.Conv3d(ch_in, ch_in, 4, 2, padding=1, bias=False)),
            BatchNorm(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=False)),
            BatchNorm(ch_out),
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
            spectral_norm(
                nn.ConvTranspose3d(
                    ch_in,
                    ch_out,
                    4,
                    2,
                    padding=1,
                    output_padding=output_padding,
                    bias=False,
                )
            ),
            BatchNorm(ch_out),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_out, ch_out, 1, 1, padding=0, bias=False)),
            BatchNorm(ch_out),
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


class PairLayer(nn.Module):
    def __init__(self, path_ref, path_target):
        super(PairLayer, self).__init__()

        self.path_ref = path_ref
        self.path_target = path_target

    def forward(self, x_ref, x_target, x_class):

        x_ref = self.path_ref(x_ref)

        if x_class is None:
            x_target = self.path_target(x_target, x_ref, x_class)

        return x_ref, x_target


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

        x_ref = x[:, self.ch_ref]
        x_target = x[:, self.ch_target]

        for ref_path, target_path in zip(self.ref_path, self.target_path):
            x_ref = checkpoint(ref_path, x_ref)
            x_target = checkpoint(target_path, x_target, x_ref, x_class)
            # x_ref = ref_path(x_ref)
            # x_target = target_path(x_target, x_ref, x_class)

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
        self.ref_bn_relu = nn.Sequential(BatchNorm(1024), nn.ReLU(inplace=True))
        self.target_bn_relu = nn.Sequential(BatchNorm(1024), nn.ReLU(inplace=True))

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

        x_class, x_ref, x_target = x_in

        x_target = self.target_fc(x_target).view(
            x_target.size()[0], 1024, self.fcsize, 1, 1
        )
        x_target = self.target_bn_relu(x_target)

        x_ref = self.ref_fc(x_ref).view(x_ref.size()[0], 1024, self.fcsize, 1, 1)
        x_ref = self.ref_bn_relu(x_ref)

        for ref_path, target_path in zip(self.ref_path, self.target_path):

            x_ref = checkpoint(ref_path, x_ref)
            x_target = checkpoint(target_path, x_target, x_ref, x_class)
            # x_ref = ref_path(x_ref)
            # x_target = target_path(x_target, x_ref, x_class)

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

        if self.noise_std > 0:
            # allocate an appropriately sized variable if it does not exist
            noise = torch.zeros(x_in.size()).float().type_as(x_in)

            # sample random noise
            noise.normal_(mean=0, std=self.noise_std)

            # add to input
            x = x_in + noise
        else:
            x = x_in

        for layer in self.path:
            x = checkpoint(layer, x)

            # x = layer(x)

        x = x.view(x.size()[0], x.shape[1] * int(self.fcsize))
        out = self.fc(x)

        return out
