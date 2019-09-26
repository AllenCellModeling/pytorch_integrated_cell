import torch.nn as nn
from ..utils import get_activation
import numpy as np


class BasicLayer(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=4, dstep=2, padding=1, activation="relu"):
        super(BasicLayer, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, ksize, dstep, padding=padding, bias=False),
            nn.BatchNorm2d(ch_out),
            get_activation(activation),
        )

    def forward(self, x):
        return self.main(x)


class BasicLayerTranspose(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        ksize=4,
        dstep=2,
        padding=0,
        output_padding=0,
        activation="relu",
    ):
        super(BasicLayerTranspose, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                ch_in,
                ch_out,
                ksize,
                dstep,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(ch_out),
            get_activation(activation),
        )

    def forward(self, x):
        return self.main(x)


class PadLayer(nn.Module):
    def __init__(self, pad_dims, pad_val=0):
        super(PadLayer, self).__init__()

        self.pad_dims = pad_dims
        self.pad_val = 0

    def forward(self, x):
        if np.sum(self.pad_dims) == 0:
            return x
        else:
            return nn.functional.pad(
                x, [0, self.pad_dims[1], 0, self.pad_dims[0]], "constant", self.pad_val
            )


class ResidLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        ksize=3,
        ch_proj_in=0,
        ch_proj_class=0,
        up_down_same="same",
        activation_last="relu",
        output_padding=0,
    ):
        super(ResidLayer, self).__init__()

        if activation_last is None:
            activation_last = "relu"

        if ksize == 1:
            padding = 0
        if ksize == 3 or ksize == 4:
            padding = 1

        if ch_in == ch_out:
            bypass_first = []
        else:
            bypass_first = nn.Conv2d(ch_in, ch_out, 1, 1, padding=0, bias=False)

        if up_down_same == "same":
            self.bypass = nn.Sequential(bypass_first)

            self.resid = nn.Sequential(
                BasicLayer(ch_in, ch_out, ksize, 1, padding=padding, activation="relu"),
                BasicLayer(
                    ch_out, ch_out, ksize, 1, padding=padding, activation="none"
                ),
            )

        elif up_down_same == "down":
            self.bypass = nn.Sequential(
                bypass_first, nn.AvgPool2d(2, stride=2, padding=0)
            )

            self.resid = nn.Sequential(
                BasicLayer(ch_in, ch_out, 4, 2, padding=1, activation="relu"),
                BasicLayer(
                    ch_out, ch_out, ksize, 1, padding=padding, activation="none"
                ),
            )

        elif up_down_same == "up":
            self.bypass = nn.Sequential(
                bypass_first, nn.Upsample(scale_factor=2), PadLayer(output_padding)
            )

            self.resid = nn.Sequential(
                BasicLayerTranspose(
                    ch_in,
                    ch_out,
                    4,
                    2,
                    padding=1,
                    output_padding=output_padding,
                    activation="relu",
                ),
                BasicLayer(
                    ch_out, ch_out, ksize, 1, padding=padding, activation="none"
                ),
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

        if x_proj is not None:
            x = x + self.proj(x_proj)

        if x_class is not None:
            x = x + self.proj_class(x_class.unsqueeze(2).unsqueeze(3)).expand(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            )

        x = self.activation(x)

        return x
