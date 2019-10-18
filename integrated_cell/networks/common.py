import torch
import torch.nn as nn
from ..utils import get_activation, spectral_norm
import numpy as np


class BasicLayer(nn.Module):
    def __init__(
        self, ch_in, ch_out, ksize=4, dstep=2, padding=1, activation="relu", bn=True
    ):
        super(BasicLayer, self).__init__()

        self.conv = spectral_norm(
            nn.Conv2d(ch_in, ch_out, ksize, dstep, padding=padding, bias=False)
        )

        if bn:
            self.bn = nn.BatchNorm2d(ch_out)
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
            nn.ConvTranspose2d(
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
            self.bn = nn.BatchNorm2d(ch_out)
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
                x, [0, self.pad_dims[1], 0, self.pad_dims[0]], "constant", 0
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
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=False)),
            nn.AvgPool2d(2, stride=2, padding=0),
        )

        self.resid = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 2, padding=1, bias=False)),
            nn.BatchNorm2d(ch_out),
            get_activation(activation),
            spectral_norm(nn.Conv2d(ch_out, ch_out, 3, 1, padding=1, bias=False)),
            nn.BatchNorm2d(ch_out),
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
            x = x + self.proj_class(x_class.unsqueeze(2).unsqueeze(3)).expand(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            )

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
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=False)),
            nn.Upsample(scale_factor=2),
            PadLayer(output_padding),
        )

        self.resid = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose2d(
                    ch_in,
                    ch_out,
                    4,
                    2,
                    padding=1,
                    output_padding=output_padding,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(ch_out),
            get_activation(activation),
            spectral_norm(nn.Conv2d(ch_out, ch_out, 3, 1, padding=1, bias=False)),
            nn.BatchNorm2d(ch_out),
        )

        self.cond_paths = nn.ModuleList([])
        for ch_cond in ch_cond_list:
            self.cond_paths.append(
                BasicLayer(ch_cond, ch_out, 1, 1, 0, activation=None)
            )

        self.activation = get_activation(activation_last)

    def forward(self, x, x_cond=[]):
        x = self.bypass(x) + self.resid(x)

        for x_c, cond_path in zip(x_cond, self.cond_paths):
            if len(x.shape) != len(x_c.shape):
                x_c = x_c.unsqueeze(2).unsqueeze(3)

            x_c = cond_path(x_c)

            if x.shape != x_c.shape:
                x_c.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

            x = x + x_c

        x = self.activation(x)

        return x


class SameLayerResidual(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        activation="relu",
        ch_proj_in=0,
        ch_proj_class=0,
        activation_last=None,
    ):
        super(SameLayerResidual, self).__init__()

        if activation_last is None:
            activation_last = activation

        self.bypass = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=False))
        )

        self.resid = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=False)),
            nn.BatchNorm2d(ch_out),
            get_activation(activation),
            spectral_norm(nn.Conv2d(ch_out, ch_out, 1, 1, padding=0, bias=False)),
            nn.BatchNorm2d(ch_out),
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
            x = x + self.proj_class(x_class.unsqueeze(2).unsqueeze(3)).expand(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            )

        x = self.activation(x)

        return x


# Adapted from https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py
class TemporalShift(nn.Module):
    def __init__(self, nth_ch_to_shift=8, n_shifts=1):
        super(TemporalShift, self).__init__()

        self.nth_ch_to_shift = nth_ch_to_shift
        self.n_shifts = n_shifts

    def forward(self, x):
        # assume inputs are B, C, H, W, Z, and we shift along Z
        n_ch_to_shift = self.nth_ch_to_shift // x.shape[1] // (2 * self.n_shifts)

        out = torch.zeros_like(x)

        for i in range(self.n_shifts):
            offset = i * 2 * n_ch_to_shift
            # shift some channels up
            out[:, : offset + n_ch_to_shift, :, :, : -(i + 1)] = x[
                :, : offset + n_ch_to_shift, :, :, (i + 1) :  # noqa
            ]

            # shift some channels down
            out[
                :,
                offset + n_ch_to_shift : offset + (2 * n_ch_to_shift),  # noqa
                :,
                :,
                (i + 1) :,  # noqa
            ] = x[
                :,
                offset + n_ch_to_shift : offset + (2 * n_ch_to_shift),  # noqa
                :,
                :,
                : -(i + 1),
            ]

        # remaining stuff we are not shifting
        out[:, offset + (2 * n_ch_to_shift) :] = x[  # noqa
            :, offset + (2 * n_ch_to_shift) :  # noqa
        ]

        return out
