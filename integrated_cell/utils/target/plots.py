import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch

from ..plots import tensor2im
from .. import utils


def summary_images(
    data_provider,
    enc,
    dec,
    mode="test",
    n_imgs_per_class=5,
    n_cols=2,
    n_rows=12,
    font_size=12,
):
    """
    Returns two images, one of sampled real data, and one of sampled generated data
    """

    font = ImageFont.truetype(
        os.path.dirname(__file__) + "/../../../etc/arial.ttf", font_size
    )

    u_classes, class_inds = np.unique(
        data_provider.get_classes(np.arange(0, data_provider.get_n_dat(mode)), mode),
        return_inverse=True,
    )

    class_names = data_provider.label_names[u_classes]
    controls = np.array(["Control - " in c for c in class_names])
    display_order_class_name = np.hstack(
        [class_names[~controls], class_names[controls]]
    )
    display_order_class_id = np.hstack([u_classes[~controls], u_classes[controls]])

    class_list = list()

    ###
    # Make the Real Images
    ###

    for c_id, c_name in zip(display_order_class_id, display_order_class_name):
        this_class_inds = np.where(class_inds == c_id)[0]
        np.random.shuffle(this_class_inds)

        class_ims, _, ref = data_provider.get_sample(
            mode, this_class_inds[0:n_imgs_per_class]
        )
        class_ims = torch.cat([ref[:, [0]], class_ims, ref[:, [1]]], 1)

        class_im = np.hstack([tensor2im([im]) for im in class_ims])

        class_im = Image.fromarray((class_im * 255).astype("uint8"))
        draw = ImageDraw.Draw(class_im)
        draw.text((20, 20), c_name, (255, 255, 255), font=font)
        class_im = np.asarray(class_im)

        class_list.append(class_im)

    rows_list = list()
    for i in range(n_cols):
        row_im = np.vstack([class_list.pop(int(i)) for i in np.zeros(n_rows)])
        rows_list.append(row_im)

    im_classes_real = np.hstack(rows_list)

    ###
    # Make the Generated Images
    ###

    class_list = list()

    im, label, ref = data_provider.get_sample(mode, np.arange(n_imgs_per_class))

    ref = ref.cuda()
    im = im.cuda()

    label_onehot = utils.index_to_onehot(label, len(u_classes)).cuda()

    with torch.no_grad():
        z = enc(im, ref, label_onehot)

    z = z[0]

    for c_id, c_name in zip(display_order_class_id, display_order_class_name):

        this_class_inds = np.where(class_inds == c_id)[0]

        _, label, _ = data_provider.get_sample(mode, this_class_inds[[0]])

        label_onehot = utils.index_to_onehot(label, len(u_classes)).cuda()

        z.normal_().cuda()
        ref = ref.cuda()

        with torch.no_grad():
            im_sampled = dec(z, ref, label_onehot)

        im_sampled = torch.cat([ref[:, [0]], im_sampled, ref[:, [1]]], 1)

        class_im = np.hstack([tensor2im([im_s]) for im_s in im_sampled])

        class_im = Image.fromarray((class_im * 255).astype("uint8"))
        draw = ImageDraw.Draw(class_im)
        draw.text((20, 20), c_name, (255, 255, 255), font=font)
        class_im = np.asarray(class_im)

        class_list.append(class_im)

    rows_list = list()
    for i in range(n_cols):
        row_im = np.vstack([class_list.pop(int(i)) for i in np.zeros(n_rows)])
        rows_list.append(row_im)

    im_classes_generated = np.hstack(rows_list)

    return im_classes_real, im_classes_generated


def pad_im(im, pad_size=2, pad_dims=[0, 1], pad_val=0.1):
    pad_dims = np.array(pad_dims)

    pad_dims = [
        [0, pad_size] if np.any(dim == pad_dims) else [0, 0] for dim in range(im.ndim)
    ]

    im = np.pad(im, pad_dims, mode="constant", constant_values=pad_val)

    return im


def summary_images_v2(
    data_provider, mode="test", n_imgs_per_class=5, classes_to_use=None, **kwargs
):

    u_classes, class_inds = np.unique(
        data_provider.get_classes(np.arange(0, data_provider.get_n_dat(mode)), mode),
        return_inverse=True,
    )
    u_class_names = data_provider.label_names[u_classes]

    gen_inds = np.random.randint(0, data_provider.get_n_dat(mode), n_imgs_per_class)

    remaining_class_inds = (
        np.sum(
            np.vstack(
                [u_class_names == class_to_use for class_to_use in classes_to_use]
            ),
            0,
        )
        == 0
    )
    remaining_classes = u_class_names[remaining_class_inds]

    imgs = summary_images_v2_aux(
        data_provider,
        classes_to_use=classes_to_use,
        mode=mode,
        n_imgs_per_class=n_imgs_per_class,
        gen_inds=gen_inds,
        **kwargs
    )
    imgs_remainder = summary_images_v2_aux(
        data_provider,
        classes_to_use=remaining_classes,
        mode=mode,
        n_imgs_per_class=n_imgs_per_class,
        gen_inds=gen_inds,
        **kwargs
    )

    return imgs, imgs_remainder


def summary_images_v2_aux(
    data_provider,
    enc,
    dec,
    mode="test",
    n_imgs_per_class=5,
    n_cols=2,
    font_size=12,
    classes_to_use=None,
    gen_inds=[],
):
    """
    Returns combined image of sampled real data, and sampled generated data
    """

    font = ImageFont.truetype(
        os.path.dirname(__file__) + "/../../../etc/arial.ttf", font_size
    )

    u_classes, class_inds = np.unique(
        data_provider.get_classes(np.arange(0, data_provider.get_n_dat(mode)), mode),
        return_inverse=True,
    )

    class_names = data_provider.label_names[u_classes]
    controls = np.array(["Control - " in c for c in class_names])

    display_order_class_name = np.hstack(
        [class_names[~controls], class_names[controls]]
    )
    display_order_class_id = np.hstack([u_classes[~controls], u_classes[controls]])

    if classes_to_use is not None:
        keep_inds = np.sum(
            np.vstack(
                [
                    display_order_class_name == class_to_use
                    for class_to_use in classes_to_use
                ]
            ),
            0,
        )
        display_order_class_name = display_order_class_name[keep_inds > 0]
        display_order_class_id = display_order_class_id[keep_inds > 0]

    im, label, ref = data_provider.get_sample(mode, np.arange(n_imgs_per_class))

    ref = ref.cuda()
    im = im.cuda()

    label_onehot = utils.index_to_onehot(label, len(u_classes)).cuda()

    with torch.no_grad():
        z = enc(im, ref, label_onehot)

    z = z[0]

    _, _, ref_gen = data_provider.get_sample(mode, gen_inds)
    ref_gen = ref_gen.cuda()

    ###
    # Make the Images
    ###

    class_list = list()

    for c_id, c_name in zip(display_order_class_id, display_order_class_name):
        # the real part
        this_class_inds = np.where(class_inds == c_id)[0]
        np.random.shuffle(this_class_inds)

        class_ims, _, ref = data_provider.get_sample(
            mode, this_class_inds[0:n_imgs_per_class]
        )
        class_ims = torch.cat([ref[:, [0]], class_ims, ref[:, [1]]], 1)

        class_im_real = [tensor2im([im]) for im in class_ims]
        class_im_real = [
            pad_im(im, pad_dims=[1]) for im in class_im_real
        ]  # pad the rhs
        class_im_real = np.hstack(class_im_real)

        # the generated part
        _, label, _ = data_provider.get_sample(mode, this_class_inds[[0]])

        label_onehot = utils.index_to_onehot(label, len(u_classes)).cuda()

        z.normal_().cuda()

        with torch.no_grad():
            im_sampled = dec(z, ref_gen, label_onehot)

        im_sampled = torch.cat([ref_gen[:, [0]], im_sampled, ref_gen[:, [1]]], 1)

        class_im_gen = [tensor2im([im]) for im in im_sampled]
        class_im_gen[:-1] = [
            pad_im(im, pad_dims=[1]) for im in class_im_gen[:-1]
        ]  # pad the rhs except for the last one
        class_im_gen = np.hstack(class_im_gen)
        # class_im_gen = np.hstack([pad_im(tensor2im([im_s])) for im_s in im_sampled])

        class_im = np.hstack(
            [pad_im(class_im_real, pad_dims=[1], pad_size=4), class_im_gen]
        )

        # the text part
        class_im = Image.fromarray((class_im * 255).astype("uint8"))
        draw = ImageDraw.Draw(class_im)
        draw.text((20, 20), c_name, (255, 255, 255), font=font)
        class_im = np.asarray(class_im)

        class_list.append(class_im)

    n_rows = int(np.ceil(len(class_list) / n_cols))

    class_im_size = class_list[0].shape

    rows_list = list()
    for i in range(n_cols):
        row_ims = [
            class_list.pop(0)
            if (len(class_list) > 0)
            else np.zeros(class_im_size).astype("uint8")
            for i in range(n_rows)
        ]
        row_ims[:-1] = [pad_im(im, pad_dims=[0], pad_val=25) for im in row_ims[:-1]]

        row_im = np.vstack(row_ims)

        rows_list.append(row_im)

    rows_list[:-1] = [
        pad_im(im, pad_dims=[1], pad_size=8, pad_val=25) for im in rows_list[:-1]
    ]

    im_out = np.hstack(rows_list)

    return im_out
