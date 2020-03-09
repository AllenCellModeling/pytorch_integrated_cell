import torch
from integrated_cell.data_providers.DataProvider import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry


class DataProvider(ParentDataProvider):
    # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
    def __init__(self, n_x_sub=4, **kwargs):

        super().__init__(**kwargs)

        self.n_x_sub = n_x_sub

    def get_sample(self, train_or_test="train", inds=None, patched=True):
        # returns
        # x         is b by c by y by x
        # x_class   is b by c by #classes
        # graph     is b by c by c - a random dag over channels

        x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)

        # this will be faster
        x = x.cuda()

        scales = 1 / (2 ** torch.arange(0, self.n_x_sub).float())

        x = [x] + [
            torch.nn.functional.interpolate(x, scale_factor=scale.item())
            for scale in scales[1:]
        ]

        return x, classes, ref
