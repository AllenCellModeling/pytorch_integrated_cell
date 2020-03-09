from integrated_cell.data_providers.DataProvider import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry


class DataProvider(ParentDataProvider):
    # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def get_sample(self, train_or_test="train", inds=None):
        # returns
        # x         is b by c by y by x
        # x_class   is b by c by #classes
        # graph     is b by c by c - a random dag over channels

        x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)

        x = x[:, [0, 2]]

        return x
