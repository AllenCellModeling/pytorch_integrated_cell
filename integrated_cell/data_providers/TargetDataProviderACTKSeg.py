from integrated_cell.data_providers.DataProviderACTK import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry


class DataProvider(ParentDataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_sample(self, train_or_test="train", inds=None):
        x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)
        
        ref = x[:, [0, 2]]
        x = x[:, [1]]

        return x, classes, ref