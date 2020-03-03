import torch

from integrated_cell.data_providers.DataProvider import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry


class DataProvider(ParentDataProvider):
    # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
    def __init__(self, channel_intensity_values=None, slice_or_proj="slice", **kwargs):

        super().__init__(**kwargs)

        self.channel_intensity_values = channel_intensity_values
        self.slice_or_proj = slice_or_proj

    def get_sample(self, train_or_test="train", inds=None):
        # returns
        # x         is b by c by y by x
        # x_class   is b by c by #classes
        # graph     is b by c by c - a random dag over channels

        x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)

        if self.channel_intensity_values:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if torch.sum(x[i, j]) > 0:
                        x[i, j] = x[i, j] * (
                            self.channel_intensity_values[j] / torch.sum(x[i, j])
                        )

        if self.slice_or_proj == "slice":
            center_of_image = torch.tensor(x.shape[2:]) / 2

            zx = x[:, :, center_of_image[0], :, :]
            zy = x[:, :, :, center_of_image[1], :]
            xy = x[:, :, :, :, center_of_image[2]]

            x = [zx, zy, xy]

        elif self.slice_or_proj == "proj":
            zx = torch.max(x, 2)[0]
            zy = torch.max(x, 3)[0]
            xy = torch.max(x, 4)[0]

            x = [zx, zy, xy]

        return x, classes, ref
