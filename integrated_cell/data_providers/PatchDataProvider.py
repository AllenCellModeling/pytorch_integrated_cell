import torch
from integrated_cell.data_providers.DataProvider import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry


def get_patch(x, patch_size):
    patch_size = torch.tensor(patch_size)

    shape_dims = torch.tensor(x.shape) - patch_size

    starts = torch.cat(
        [torch.randint(i, [1]) if i > 0 else torch.tensor([0]) for i in shape_dims], 0
    )
    ends = starts + patch_size

    slices = tuple(slice(s, e) for s, e in zip(starts, ends))

    x = x[slices]

    return x


class DataProvider(ParentDataProvider):
    # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
    def __init__(self, patch_size=[3, 64, 64, 32], **kwargs):

        super().__init__(**kwargs)

        self.patch_size = patch_size

    def get_sample(self, train_or_test="train", inds=None, patched=True):
        # returns
        # x         is b by c by y by x
        # x_class   is b by c by #classes
        # graph     is b by c by c - a random dag over channels

        x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)

        if patched:
            x = torch.stack([get_patch(x_sub, self.patch_size) for x_sub in x], 0)

        return x, classes, ref
