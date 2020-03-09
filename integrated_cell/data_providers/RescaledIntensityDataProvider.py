import torch

from integrated_cell.data_providers.DataProvider import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry


class DataProvider(ParentDataProvider):
    # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
    def __init__(
        self, channel_intensity_values=[2596.9521, 2596.9521, 2596.9521], **kwargs
    ):

        super().__init__(**kwargs)

        self.channel_intensity_values = channel_intensity_values

    def get_sample(self, train_or_test="train", inds=None):
        # returns
        # x         is b by c by y by x
        # x_class   is b by c by #classes
        # graph     is b by c by c - a random dag over channels

        x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if torch.sum(x[i, j]) > 0:
                    x[i, j] = x[i, j] * (
                        self.channel_intensity_values[j] / torch.sum(x[i, j])
                    )

        return x, classes, ref
