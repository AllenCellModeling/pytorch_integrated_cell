import torch
from integrated_cell.data_providers.DataProvider import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry


class DataProvider(ParentDataProvider):
    # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
    def __init__(self, masked_channels=[], **kwargs):

        super().__init__(**kwargs)

        self.masked_channels = masked_channels

    def get_sample(self, train_or_test="train", inds=None):

        x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)

        # build a mask for channels
        n_channels = x.shape[1]
        channel_mask = torch.zeros(n_channels).byte()
        channel_mask[self.masked_channels] = 1

        x[:, channel_mask] = 0

        return x, classes, ref
