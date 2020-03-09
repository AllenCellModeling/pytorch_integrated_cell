import numpy as np
import torch


def crop_to(images, crop_to):
    crop = (np.array(images.shape[2:]) - np.array(crop_to)) / 2
    crop_pre = np.floor(crop).astype(int)
    crop_post = np.ceil(crop).astype(int)

    pad_pre = -crop_pre
    pad_pre[pad_pre < 0] = 0

    pad_post = -crop_post
    pad_post[pad_post < 0] = 0

    crop_pre[crop_pre < 0] = 0

    crop_post[crop_post < 0] = 0
    crop_post[crop_post == 0] = -np.array(images.shape[2:])[crop_post == 0]

    if len(crop_pre) == 2:
        images = images[
            :,
            :,
            crop_pre[0] : -crop_post[0],  # noqa
            crop_pre[1] : -crop_post[1],  # noqa
        ]

    elif len(crop_pre) == 3:
        images = images[
            :,
            :,
            crop_pre[0] : -crop_post[0],  # noqa
            crop_pre[1] : -crop_post[1],  # noqa
            crop_pre[2] : -crop_post[2],  # noqa
        ]

    pad_pre = np.hstack([np.zeros(2), pad_pre])
    pad_post = np.hstack([np.zeros(2), pad_post])

    padding = np.vstack([pad_pre, pad_post]).transpose().astype("int")

    images = np.pad(images, padding, mode="constant", constant_values=0)

    images = torch.tensor(images)
