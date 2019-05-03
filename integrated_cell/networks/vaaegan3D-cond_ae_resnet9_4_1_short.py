from torch import nn
import torch
import numpy as np

from ..utils import spectral_norm
from ..utils import get_activation

# decoder are DCGAN with spectral norm
# like 9_4 but with 3x3 convolutions on the 2nd half of the residual path


class BasicLayer(nn.Module):
    def __init__(
        self, ch_in, ch_out, ksize=4, dstep=2, padding=1, activation="relu", bn=True
    ):
        super(BasicLayer, self).__init__()

        self.conv = spectral_norm(
            nn.Conv3d(ch_in, ch_out, ksize, dstep, padding=padding, bias=False)
        )

        if bn:
            self.bn = nn.BatchNorm3d(ch_out)
        else:
            self.bn = None

        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        x = self.activation(x)

        return x


class BasicLayerTranspose(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        ksize=4,
        dstep=2,
        padding=1,
        output_padding=0,
        activation="relu",
        bn=True,
    ):
        super(BasicLayerTranspose, self).__init__()

        self.conv = spectral_norm(
            nn.ConvTranspose3d(
                ch_in,
                ch_out,
                ksize,
                dstep,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            )
        )
        if bn:
            self.bn = nn.BatchNorm3d(ch_out)
        else:
            self.bn = None

        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
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


class DownLayerResidual(nn.Module):
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
        super(DownLayerResidual, self).__init__()

        if activation_last is None:
            activation_last = activation

        self.bypass = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=0),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=False)),
        )

        self.resid = nn.Sequential(
            spectral_norm(nn.Conv3d(ch_in, ch_in, 4, 2, padding=1, bias=False)),
            nn.BatchNorm3d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=False)),
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


class SameLayerResidual(nn.Module):
    def __init__(
        self,
        ch_in,
        ksize=4,
        dstep=2,
        activation="relu",
        ch_proj_in=0,
        ch_proj_class=0,
        activation_last=None,
    ):
        super(SameLayerResidual, self).__init__()

        ch_out = ch_in

        if activation_last is None:
            activation_last = activation

        self.bypass = nn.Sequential()

        self.resid = nn.Sequential(
            spectral_norm(nn.Conv3d(ch_in, ch_in, 3, 1, padding=1, bias=False)),
            nn.BatchNorm3d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=False)),
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


class UpLayerResidual(nn.Module):
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
        super(UpLayerResidual, self).__init__()

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
                    ch_in,
                    4,
                    2,
                    padding=1,
                    output_padding=output_padding,
                    bias=False,
                )
            ),
            nn.BatchNorm3d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=False)),
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
        l_sizes=[64, 128, 256, 512, 512],
        imsize_compressed=[5, 3, 2],
    ):
        super(Enc, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.ch_ref = ch_ref
        self.ch_target = ch_target

        self.n_ref = n_ref
        self.n_latent_dim = n_latent_dim

        self.l_sizes = l_sizes
        self.imsize_compressed = imsize_compressed

        self.ref_path = nn.ModuleList([DownLayerResidual(n_channels, l_sizes[0])])

        self.target_path = nn.ModuleList(
            [
                DownLayerResidual(
                    n_channels_target,
                    l_sizes[0],
                    ch_proj_in=l_sizes[0],
                    ch_proj_class=n_classes,
                )
            ]
        )

        for i in range(len(l_sizes) - 1):
            self.ref_path.append(DownLayerResidual(l_sizes[i], l_sizes[i + 1]))
            self.target_path.append(
                DownLayerResidual(
                    l_sizes[i],
                    l_sizes[i + 1],
                    ch_proj_in=l_sizes[i + 1],
                    ch_proj_class=n_classes,
                )
            )

        if self.n_ref > 0:
            self.ref_out_mu = nn.Sequential(
                nn.Linear(
                    l_sizes[-1] * int(np.prod(self.imsize_compressed)),
                    self.n_ref,
                    bias=False,
                )
            )

            self.ref_out_sigma = nn.Sequential(
                nn.Linear(
                    l_sizes[-1] * int(np.prod(self.imsize_compressed)),
                    self.n_ref,
                    bias=False,
                )
            )

        if self.n_latent_dim > 0:
            self.latent_out_mu = nn.Sequential(
                nn.Linear(
                    l_sizes[-1] * int(np.prod(self.imsize_compressed)),
                    self.n_latent_dim,
                    bias=False,
                )
            )

            self.latent_out_sigma = nn.Sequential(
                nn.Linear(
                    l_sizes[-1] * int(np.prod(self.imsize_compressed)),
                    self.n_latent_dim,
                    bias=False,
                )
            )

    def forward(self, x, x_class):
        x_ref = x[:, self.ch_ref]
        x_target = x[:, self.ch_target]

        for ref_path, target_path in zip(self.ref_path, self.target_path):
            x_ref = ref_path(x_ref)
            x_target = target_path(x_target, x_ref, x_class)

        x_ref = x_ref.view(x_ref.size()[0], -1)
        x_target = x_target.view(x_target.size()[0], -1)

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
        padding_latent=(0, 0, 0),  # (1,1,0)
        ch_ref=[0, 2],
        ch_target=[1],
        l_sizes=[512, 512, 256, 128, 64],
        imsize_compressed=[5, 3, 2],
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

        self.l_sizes = l_sizes
        self.imsize_compressed = imsize_compressed

        self.ref_fc = nn.Linear(
            self.n_ref, l_sizes[0] * int(np.prod(self.imsize_compressed)), bias=False
        )
        self.target_fc = nn.Linear(
            self.n_latent_dim,
            l_sizes[0] * int(np.prod(self.imsize_compressed)),
            bias=False,
        )
        self.ref_bn_relu = nn.Sequential(
            nn.BatchNorm3d(l_sizes[0]), nn.ReLU(inplace=True)
        )
        self.target_bn_relu = nn.Sequential(
            nn.BatchNorm3d(l_sizes[0]), nn.ReLU(inplace=True)
        )

        self.ref_path = nn.ModuleList([])
        self.target_path = nn.ModuleList([])

        for i in range(len(l_sizes) - 1):
            if i == 0:
                padding = padding_latent
            else:
                padding = 0

            self.ref_path.append(
                UpLayerResidual(l_sizes[i], l_sizes[i + 1], output_padding=padding)
            )
            self.target_path.append(
                UpLayerResidual(
                    l_sizes[i],
                    l_sizes[i + 1],
                    output_padding=padding,
                    ch_proj_in=l_sizes[i + 1],
                    ch_proj_class=n_classes,
                )
            )

        self.ref_path.append(
            UpLayerResidual(l_sizes[i + 1], n_channels, activation_last="sigmoid")
        )
        self.target_path.append(
            UpLayerResidual(
                l_sizes[i + 1],
                n_channels_target,
                ch_proj_in=n_channels,
                ch_proj_class=n_classes,
                activation_last="sigmoid",
            )
        )

    def forward(self, x_in):
        x_class, x_ref, x_target = x_in

        x_target = self.target_fc(x_target).view(
            x_target.size()[0],
            self.l_sizes[0],
            self.imsize_compressed[0],
            self.imsize_compressed[1],
            self.imsize_compressed[2],
        )

        x_target = self.target_bn_relu(x_target)

        x_ref = self.ref_fc(x_ref).view(
            x_ref.size()[0],
            self.l_sizes[0],
            self.imsize_compressed[0],
            self.imsize_compressed[1],
            self.imsize_compressed[2],
        )
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
    def __init__(
        self,
        n_classes,
        n_channels,
        gpu_ids,
        ch_proj_class=0,
        noise_std=0,
        l_sizes=[64, 128, 256, 512, 1024, 1024],
        activation="leakyrelu",
        **kwargs
    ):
        super(DecD, self).__init__()

        self.noise_std = noise_std

        self.noise = torch.zeros(0)

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.noise = torch.zeros(0)

        self.path = nn.ModuleList([])

        self.l_sizes = l_sizes

        self.path.append(
            DownLayerResidual(
                n_channels,
                self.l_sizes[0],
                ch_proj_class=ch_proj_class,
                activation=activation,
            )
        )

        for i in range(len(l_sizes) - 1):
            self.path.append(
                DownLayerResidual(
                    self.l_sizes[i],
                    self.l_sizes[i + 1],
                    ch_proj_class=ch_proj_class,
                    activation=activation,
                )
            )

        self.fc = spectral_norm(
            nn.Linear(self.l_sizes[-1] * int(self.fcsize), n_classes, bias=False)
        )

    def forward(self, x_in, x_class=None):
        gpu_ids = self.gpu_ids

        if self.noise_std > 0:
            # allocate an appropriately sized variable if it does not exist
            if self.noise.size() != x_in.size():
                self.noise = torch.zeros(x_in.size()).cuda(gpu_ids[0])

            # sample random noise
            self.noise.normal_(mean=0, std=self.noise_std)
            noise = torch.autograd.Variable(self.noise)

            # add to input
            x = x_in + noise
        else:
            x = x_in

        for layer in self.path:
            x = layer(x, x_class=x_class)

        x = x.view(x.size()[0], x.shape[1] * int(self.fcsize))
        out = self.fc(x)

        return out
