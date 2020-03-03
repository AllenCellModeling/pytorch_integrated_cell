from torch import nn
import torch
from ..utils import spectral_norm

from ..utils import get_activation
import numpy as np

# from torch.utils.checkpoint import checkpoint
# like vaaegan2D-cond_ae_resnet9_4_cagan but with definable channel numbers on layers
# and 3D

# class ResidualLayer(nn.Module):
#     def __init__(
#         self, ch_in, ch_out, ksize=4, dstep=2, padding=1, activation="relu", bn=True
#     ):
#         super(ResidualLayer, self).__init__()

#         self.main = nn.Sequential(
#             spectral_norm(
#                 nn.Conv3d(ch_in, ch_out, ksize, dstep, padding=padding, bias=True)
#             ),
#             nn.BatchNorm3d(ch_out),
#             get_activation(activation),
#             spectral_norm(nn.Conv3d(ch_out, ch_out, 3, 1, padding=1, bias=True)),
#             nn.BatchNorm3d(ch_out),
#         )

#     def forward(self, x):
#         x = self.main(x)

#         return x


class BasicLayer(nn.Module):
    def __init__(
        self, ch_in, ch_out, ksize=4, dstep=2, padding=1, activation="relu", bn=True
    ):
        super(BasicLayer, self).__init__()

        self.conv = spectral_norm(
            nn.Conv3d(ch_in, ch_out, ksize, dstep, padding=padding, bias=True)
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
                bias=True,
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
            spectral_norm(nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=True)),
        )

        self.resid = nn.Sequential(
            spectral_norm(nn.Conv3d(ch_in, ch_in, 4, 2, padding=1, bias=True)),
            nn.BatchNorm3d(ch_in),
            get_activation(activation),
            spectral_norm(nn.Conv3d(ch_in, ch_out, 3, 1, padding=1, bias=True)),
            nn.BatchNorm3d(ch_out),
        )

        self.proj = None
        if ch_proj_in > 0:
            self.proj = BasicLayer(ch_proj_in, ch_out, 1, 1, 0)

        self.proj_class = None
        if ch_proj_class > 0:
            self.proj_class = BasicLayer(ch_proj_class, ch_out, 1, 1, 0)

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
        ch_cond_list=[],
        activation_last=None,
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

        self.cond_paths = nn.ModuleList([])
        for ch_cond in ch_cond_list:
            self.cond_paths.append(BasicLayer(ch_cond, ch_out, 1, 1, 0))

        self.activation = get_activation(activation_last)

    def forward(self, x, *x_cond):
        x = self.bypass(x) + self.resid(x)

        for x_c, cond_path in zip(x_cond, self.cond_paths):
            if len(x.shape) != len(x_c.shape):
                x_c = x_c.unsqueeze(2).unsqueeze(3).unsqueeze(4)

            x_c = cond_path(x_c)

            if x.shape != x_c.shape:
                x_c.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])

            x = x + x_c

        x = self.activation(x)

        return x


class Enc(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        n_classes,
        gpu_ids,
        ch_ref=[0, 2],
        ch_target=[1],
        conv_channels_list=[32, 64, 128, 256, 512],
        noise_std=0,
        imsize_compressed=[5, 3, 2],
    ):
        super(Enc, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.ch_ref = ch_ref
        self.ch_target = ch_target

        self.n_latent_dim = n_latent_dim

        self.noise_std = noise_std
        self.noise = torch.zeros(0)

        self.target_path = nn.ModuleList(
            [
                DownLayerResidual(
                    len(ch_target),
                    conv_channels_list[0],
                    ch_proj_in=len(ch_ref),
                    ch_proj_class=n_classes,
                )
            ]
        )

        for ch_in, ch_out in zip(conv_channels_list[0:-1], conv_channels_list[1:]):
            self.target_path.append(
                DownLayerResidual(
                    ch_in, ch_out, ch_proj_in=len(ch_ref), ch_proj_class=n_classes
                )
            )

            ch_in = ch_out

        if self.n_latent_dim > 0:
            self.latent_out_mu = spectral_norm(
                nn.Linear(
                    ch_in * int(np.prod(imsize_compressed)),
                    self.n_latent_dim,
                    bias=True,
                )
            )

            self.latent_out_sigma = spectral_norm(
                nn.Linear(
                    ch_in * int(np.prod(imsize_compressed)),
                    self.n_latent_dim,
                    bias=True,
                )
            )

    def forward(self, x_target, x_ref=None, x_class=None):

        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        scales = 1 / (2 ** torch.arange(0, len(self.target_path) + 1).float())

        if x_ref is None:
            x_ref = [None] * len(scales)
        else:
            x_ref = [
                torch.nn.functional.interpolate(x_ref, scale_factor=scale.item())
                for scale in scales[1:]
            ]

        for ref, target_path in zip(x_ref, self.target_path):
            x_target = target_path(x_target, ref, x_class)

        x_target = x_target.view(x_target.size()[0], -1)

        mu = self.latent_out_mu(x_target)
        logsigma = self.latent_out_sigma(x_target)

        return [mu, logsigma]


class Dec(nn.Module):
    def __init__(
        self,
        n_latent_dim,
        n_classes,
        gpu_ids,
        padding_latent=[0, 0, 0],
        imsize_compressed=[5, 3, 2],
        pretrained_path=None,
        ch_ref=[0, 2],
        ch_target=[1],
        conv_channels_list=[512, 256, 128, 64, 32],
        proj_z=False,
        proj_z_ref_to_target=False,
        activation_last="sigmoid",
    ):

        super(Dec, self).__init__()

        self.gpu_ids = gpu_ids
        self.padding_latent = padding_latent
        self.imsize_compressed = imsize_compressed

        self.ch_first = conv_channels_list[0]

        self.n_latent_dim = n_latent_dim
        self.n_classes = n_classes

        self.n_channels = len(ch_target)

        self.ch_ref = ch_ref
        self.ch_target = ch_target

        self.proj_z = proj_z
        self.proj_z_ref_to_target = proj_z_ref_to_target

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

        target_cond_list = []
        if len(ch_ref) > 0:
            target_cond_list.append(len(ch_ref))

        if n_classes > 0:
            target_cond_list.append(n_classes)

        l_sizes = conv_channels_list
        for i in range(len(l_sizes) - 1):
            if i == 0:
                padding = padding_latent
            else:
                padding = 0

            self.target_path.append(
                UpLayerResidual(
                    l_sizes[i],
                    l_sizes[i + 1],
                    output_padding=padding,
                    ch_cond_list=target_cond_list,
                )
            )

        self.target_path.append(
            UpLayerResidual(
                l_sizes[i + 1],
                len(ch_target),
                ch_cond_list=target_cond_list,
                activation_last=activation_last,
            )
        )

    def forward(self, z_target, x_ref=None, x_class=None):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:

        scales = 1 / (2 ** torch.arange(0, len(self.target_path)).float())

        if x_ref is None:
            x_ref = [None] * (len(scales) + 1)
        else:
            x_ref = [x_ref] + [
                torch.nn.functional.interpolate(x_ref, scale_factor=scale.item())
                for scale in scales[1:]
            ]
            x_ref = x_ref[::-1]

        x_target = self.target_fc(z_target).view(
            z_target.size()[0],
            self.ch_first,
            self.imsize_compressed[0],
            self.imsize_compressed[1],
            self.imsize_compressed[2],
        )
        x_target = self.target_bn_relu(x_target)

        for ref, target_path in zip(x_ref, self.target_path):

            target_cond = [ref, x_class]

            x_target = target_path(x_target, *target_cond)

        return x_target


class DecD(nn.Module):
    def __init__(
        self,
        n_classes,
        n_channels,
        gpu_ids,
        noise_std,
        activation="leakyrelu",
        n_classes_out=1,
        conv_channels_list=[32, 64, 128, 256, 512],
        imsize_compressed=[5, 3, 2],
        **kwargs
    ):
        super(DecD, self).__init__()

        self.noise_std = noise_std
        self.noise = torch.zeros(0)

        self.gpu_ids = gpu_ids
        self.fcsize = 2
        self.imsize_compressed = imsize_compressed

        self.path = nn.ModuleList([])

        l_sizes = [n_channels] + conv_channels_list
        for i in range(len(l_sizes) - 1):
            self.path.append(
                DownLayerResidual(
                    l_sizes[i],
                    l_sizes[i + 1],
                    ch_proj_class=n_classes,
                    activation="leakyrelu",
                )
            )

        self.fc = spectral_norm(
            nn.Linear(
                conv_channels_list[-1] * int(np.prod(imsize_compressed)),
                n_classes_out,
                bias=True,
            )
        )

    def forward(self, x_in, y_in=None):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:

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
            x = layer(x, x_class=y_in)

        x = x.view(x.size()[0], x.shape[1] * int(np.prod(self.imsize_compressed)))

        out = self.fc(x).squeeze()

        return out
