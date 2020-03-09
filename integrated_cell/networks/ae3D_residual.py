from torch import nn
import numpy as np

from ..utils import spectral_norm
from ..utils import get_activation

# An autoencoder for 3D images using modified residual layers
# This is effectively a simplified version of vaegan3D_cgan_target2.py


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
    def __init__(self, ch_in, ch_out, activation="relu", activation_last=None):
        super(DownLayerResidual, self).__init__()

        if activation_last is None:
            activation_last = activation

        self.bypass = nn.Sequential(
            nn.AvgPool3d(2, stride=2, padding=0),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
        )

        self.resid = nn.Sequential(
            spectral_norm(nn.Conv3d(ch_in, ch_in, 4, 2, padding=1, bias=True)),
            nn.BatchNorm3d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=True)),
            nn.BatchNorm3d(ch_out),
        )

        self.activation = get_activation(activation_last)

    def forward(self, x, x_proj=None, x_class=None):

        x = self.bypass(x) + self.resid(x)

        x = self.activation(x)

        return x


class UpLayerResidual(nn.Module):
    def __init__(
        self, ch_in, ch_out, activation="relu", output_padding=0, activation_last=None
    ):
        super(UpLayerResidual, self).__init__()

        if activation_last is None:
            activation_last = activation

        self.bypass = nn.Sequential(
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
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
                    bias=True,
                )
            ),
            nn.BatchNorm3d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=True)),
            nn.BatchNorm3d(ch_out),
        )

        self.activation = get_activation(activation_last)

    def forward(self, x):
        x = self.bypass(x) + self.resid(x)

        x = self.activation(x)

        return x


class Enc(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        gpu_ids,
        n_ch=3,
        conv_channels_list=[32, 64, 128, 256, 512],
        imsize_compressed=[5, 3, 2],
    ):
        super(Enc, self).__init__()

        self.gpu_ids = gpu_ids
        self.n_latent_dim = n_latent_dim

        self.target_path = nn.ModuleList(
            [DownLayerResidual(n_ch, conv_channels_list[0])]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(DownLayerResidual(ch_in, ch_out))

            ch_in = ch_out

        self.latent_out = spectral_norm(
            nn.Linear(
                ch_in * int(np.prod(imsize_compressed)), self.n_latent_dim, bias=True
            )
        )

    def forward(self, x_target):

        for target_path in self.target_path:
            x_target = target_path(x_target)

        x_target = x_target.view(x_target.size()[0], -1)

        z = self.latent_out(x_target)

        return z


class Dec(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        gpu_ids,
        padding_latent=[0, 0, 0],
        imsize_compressed=[5, 3, 2],
        n_ch=3,
        conv_channels_list=[512, 256, 128, 64, 32],
        activation_last="sigmoid",
    ):

        super(Dec, self).__init__()

        self.gpu_ids = gpu_ids
        self.padding_latent = padding_latent
        self.imsize_compressed = imsize_compressed

        self.ch_first = conv_channels_list[0]

        self.n_latent_dim = n_latent_dim

        self.n_channels = n_ch

        self.target_fc = spectral_norm(
            nn.Linear(
                self.n_latent_dim,
                conv_channels_list[0] * int(np.prod(self.imsize_compressed)),
                bias=True,
            )
        )

        self.target_bn_relu = nn.Sequential(
            nn.BatchNorm3d(conv_channels_list[0]), nn.ReLU(inplace=True)
        )

        self.target_path = nn.ModuleList([])

        l_sizes = conv_channels_list
        for i in range(len(l_sizes) - 1):
            if i == 0:
                padding = padding_latent
            else:
                padding = 0

            self.target_path.append(
                UpLayerResidual(l_sizes[i], l_sizes[i + 1], output_padding=padding)
            )

        self.target_path.append(
            UpLayerResidual(l_sizes[i + 1], n_ch, activation_last=activation_last)
        )

    def forward(self, z_target):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:

        x_target = self.target_fc(z_target).view(
            z_target.size()[0],
            self.ch_first,
            self.imsize_compressed[0],
            self.imsize_compressed[1],
            self.imsize_compressed[2],
        )
        x_target = self.target_bn_relu(x_target)

        for target_path in self.target_path:

            x_target = target_path(x_target)

        return x_target
