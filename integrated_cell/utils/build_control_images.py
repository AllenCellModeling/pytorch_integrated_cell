import tifffile
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import scipy
from scipy.stats import norm

from .utils import str2rand
from .imgToProjection import imgtoprojection


# I mostly copied over the parts from the IPP to clone the processing to make these control images.
# This should be moved into the IPP later - GRJ 05/2019


def build_control_images(
    save_dir,
    csv_path,
    image_parent,
    split_seed=615,
    verbose=True,
    target_col="StructureId/Name",
):
    # constructs "control" images that contain the following content as the "structure" channel:
    # DNA label
    # Membrane label
    # Random structure from a different cell
    # Blank (all zeros)
    # Poisson noise

    # make a dataframe out of the csv log file
    if verbose:
        print("reading csv manifest")
    csv_df = pd.read_csv(csv_path)

    image_col = "save_reg_path"

    csv_df["ch_memb"] = 3
    csv_df["ch_struct"] = 4
    csv_df["ch_dna"] = 2
    csv_df["ch_seg_cell"] = 1
    csv_df["ch_seg_nuc"] = 0
    csv_df["ch_trans"] = 5

    ch_names = ["nuc", "cell", "dna", "memb", "struct", "trans"]

    im_paths = list()

    for i, im_path in enumerate(csv_df[image_col]):
        splits = np.array(im_path.split("/"))
        lens = np.array([len(s) for s in splits])
        splits = splits[lens > 0]
        im_paths += ["/".join(splits[-2::])]

    csv_df[image_col] = im_paths

    # csv_df = check_files(csv_df, image_col, verbose)

    # pick some cells at random
    image_classes = list(csv_df[target_col])
    [label_names, labels] = np.unique(image_classes, return_inverse=True)

    n_classes = len(label_names)
    fract_sample_labels = 1 / n_classes

    rand = str2rand(csv_df["CellId"], split_seed)
    rand_struct = np.argsort(str2rand(csv_df["CellId"], split_seed + 1))

    duplicate_inds = rand <= fract_sample_labels

    df_controls = csv_df[duplicate_inds].copy().reset_index()

    random_scale = norm.ppf(0.999)

    channels_to_make = ["DNA", "Memb", "Blank", "Noise", "Random"]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    im = load_image(csv_df.iloc[0], image_parent, image_col)
    n_ch = im.shape[0]

    for index in tqdm(range(len(df_controls))):
        row = df_controls.iloc[index]

        save_path = "{}/control_{}.tiff".format(save_dir, index)

        save_path_flats = [
            "{}/control_{}_{}_flat.png".format(save_dir, index, ch_name)
            for ch_name in channels_to_make
        ]
        save_path_flat_projs = [
            "{}/control_{}_{}_flat_proj.png".format(save_dir, index, ch_name)
            for ch_name in channels_to_make
        ]

        try:
            im = load_image(row, image_parent, image_col)
        except:  # noqa
            print("Skipping image {}".format(row[image_col]))
            continue

        n_ch = im.shape[0]

        ch_dna = im[row["ch_dna"]] * (im[row["ch_seg_nuc"]] > 0)
        ch_memb = im[row["ch_memb"]]

        ch_blank = np.zeros(im[0].shape)

        ch_noise = (np.random.normal(0, 1, im[0].shape) / random_scale) * 255
        ch_noise[ch_noise < 0] = 0
        ch_noise[ch_noise > 255] = 255

        random_image_row = csv_df.iloc[rand_struct[index]]
        ch_random = load_image(random_image_row, image_parent, image_col)[
            random_image_row["ch_struct"]
        ]

        im_out = np.concatenate(
            [im]
            + [
                np.expand_dims(ch, 0)
                for ch in [ch_dna, ch_memb, ch_blank, ch_noise, ch_random]
            ],
            0,
        )

        im_tmp = im_out.astype("uint8")
        tifffile.imsave(save_path, im_tmp)

        im_crop = crop_cell_nuc(im_out, ch_names + channels_to_make)

        row_list = list()
        for i, channel in enumerate(channels_to_make):
            name = "Control - {}".format(channel)

            row = row.copy()

            row["StructureDisplayName"] = name
            row["StructureId"] = -i
            row["StructureShortName"] = name
            row["ProteinId/Name"] = name
            row["StructureId/Name"] = name

            row["save_reg_path"] = save_path
            row["save_reg_path_flat"] = save_path_flats[i]
            row["save_reg_path_flat_proj"] = save_path_flat_projs[i]

            row["ch_struct"] = n_ch + i

            row_list.append(row)

            im_tmp = im_crop[[2, 3, n_ch + i]]
            colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

            im_flat = imgtoprojection(
                im_tmp / 255,
                proj_all=True,
                proj_method="max",
                local_adjust=False,
                global_adjust=True,
                colors=colors,
            )
            scipy.misc.imsave(
                row["save_reg_path_flat_proj"], im_flat.transpose(1, 2, 0)
            )

            im_flat = imgtoprojection(
                im_tmp / 255,
                proj_all=False,
                proj_method="max",
                local_adjust=False,
                global_adjust=True,
                colors=colors,
            )
            scipy.misc.imsave(row["save_reg_path_flat"], im_flat.transpose(1, 2, 0))

        csv_df = csv_df.append(row_list)

    csv_df.to_csv("{}/data_plus_controls.csv".format(save_dir))


def check_files(csv_df, image_col="save_reg_path", verbose=True):
    # TODO add checking to make sure number of keys in h5 file matches number of lines in csv file
    if verbose:
        print("Checking the existence of files")

    for index in tqdm(range(0, csv_df.shape[0]), desc="Checking files", ascii=True):
        is_good_row = True

        row = csv_df.loc[index]

        image_path = os.sep + row[image_col]

        try:
            load_image(row)
        except:  # noqa
            print("Could not load from image. " + image_path)
            is_good_row = False

        csv_df.loc[index, "valid_row"] = is_good_row

    # only work with valid rows
    n_old_rows = len(csv_df)
    csv_df = csv_df.loc[csv_df["valid_row"] == True]  # noqa
    csv_df = csv_df.drop("valid_row", 1)
    n_new_rows = len(csv_df)
    if verbose:
        print("{0}/{1} samples have all files present".format(n_new_rows, n_old_rows))

    return csv_df


def load_image(df_row, image_parent, image_col):

    im_path = image_parent + os.sep + df_row[image_col]

    im = tifffile.imread(im_path)

    return im


def crop_cell_nuc(im, channel_names):

    nuc_ind = np.array(channel_names) == "nuc"
    cell_ind = np.array(channel_names) == "cell"

    dna_ind = np.array(channel_names) == "dna"
    # trans_ind = np.array(channel_names) == "trans"

    other_channel_inds = np.ones(len(channel_names))
    other_channel_inds[nuc_ind | cell_ind | dna_ind] = 0

    im[dna_ind] = im[dna_ind] * im[nuc_ind]

    for i in np.where(other_channel_inds)[0]:
        im[i] = im[i] * im[cell_ind]

    return im
