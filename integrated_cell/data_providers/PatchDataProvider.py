import torch
from integrated_cell.data_providers.DataProvider import (
    DataProvider as ParentDataProvider,
)  # ugh im sorry


def get_patch_slice(shape, patch_size):
    patch_size = torch.tensor(patch_size)

    shape_dims = torch.tensor(shape) - patch_size

    starts = torch.cat(
        [torch.randint(i, [1]) if i > 0 else torch.tensor([0]) for i in shape_dims], 0
    )
    ends = starts + patch_size

    slices = tuple(slice(s, e) for s, e in zip(starts, ends))

    # x = x[slices]

    return slices


class DataProvider(ParentDataProvider):
    # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
    def __init__(self, patch_size=[3, 64, 64, 32], default_return_mesh=False, **kwargs):

        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.default_return_mesh = default_return_mesh

    def get_sample(
        self, train_or_test="train", inds=None, patched=True, return_mesh=None
    ):
        # returns
        # x         is b by c by y by x
        # x_class   is b by c by #classes
        # graph     is b by c by c - a random dag over channels

        if return_mesh is None:
            return_mesh = self.default_return_mesh

        x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)

        shape = x[0].shape

        if return_mesh:

            meshes = [torch.arange(0, i) - (i // 2) for i in list(shape[1:])]
            mesh = torch.meshgrid(meshes)
            mesh = torch.stack([m.float() for m in mesh], 0)

            if not patched:
                ref = torch.stack([mesh for i in range(x.shape[0])], 0)

        if patched:

            slices = [
                get_patch_slice(shape, self.patch_size) for i in range(x.shape[0])
            ]

            x = torch.stack(
                [x_sub[slices_sub] for x_sub, slices_sub in zip(x, slices)], 0
            )

            if return_mesh:
                ref = torch.stack(
                    [
                        mesh[[slice(0, mesh.shape[0])] + list(slices[0][1:])]
                        for slices_sub in slices
                    ],
                    0,
                )

        return x, classes, ref
