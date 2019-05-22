import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import tifffile
from PIL import Image
import hashlib

from integrated_cell.data_providers.DataProviderABC import DataProviderABC


def str2rand(strings, seed):
    # prepend the 'seed' to the unique file path, then hash with SHA512
    salted_string = [str(seed) + str(string) for string in strings]
    hash_strings = [
        hashlib.sha512(string.encode("utf-8")).hexdigest() for string in salted_string
    ]

    rand_nums = list()
    # Pull out the first 5 digits to get a value between 0-1 inclusive
    for hash_string in hash_strings:
        str_nums = [char for pos, char in enumerate(hash_string) if char.isdigit()]
        str_num = "".join(str_nums[0:5])
        num = float(str_num) / 100000
        rand_nums.append(num)

    rand_nums = np.array(rand_nums)

    return rand_nums


class DataProvider(DataProviderABC):
    def __init__(
        self,
        image_parent,
        batch_size,
        n_dat=-1,
        csv_name="data_jobs_out.csv",
        hold_out=0.1,
        verbose=True,
        target_col="StructureId/Name",
        channelInds=[0, 1, 2],
        check_files=True,
        split_seed=1,
        crop_to=None,
        return2D=False,
        rescale_to=None,
    ):

        self.data = {}

        self.hold_out = hold_out
        self.verbose = verbose
        self.target_col = target_col
        self.channelInds = np.array(channelInds)
        self.check_files = check_files
        self.split_seed = split_seed
        self.crop_to = crop_to
        self.rescale_to = rescale_to
        self.image_parent = image_parent
        self.csv_name = csv_name

        self.return2D = return2D

        self.channel_dict = {
            0: "ch_memb",
            1: "ch_struct",
            2: "ch_dna",
            3: "ch_seg_cell",
            4: "ch_seg_nuc",
            5: "ch_trans",
        }

        # channel_column_list = [col for col in self.channel_index_to_column_dict.values()]

        # make a dataframe out of the csv log file
        csv_path = os.path.join(self.image_parent, self.csv_name)
        if self.verbose:
            print("reading csv manifest")
        csv_df = pd.read_csv(csv_path)

        if return2D:
            self.image_col = "save_reg_path_flat"

            csv_df["ch_memb"] = 0
            csv_df["ch_struct"] = 1
            csv_df["ch_dna"] = 2
        else:
            self.image_col = "save_reg_path"

            csv_df["ch_memb"] = 3
            csv_df["ch_struct"] = 4
            csv_df["ch_dna"] = 2
            csv_df["ch_seg_cell"] = 1
            csv_df["ch_seg_nuc"] = 0
            csv_df["ch_trans"] = 5

        im_paths = list()

        for i, im_path in enumerate(csv_df[self.image_col]):
            splits = np.array(im_path.split("/"))
            lens = np.array([len(s) for s in splits])
            splits = splits[lens > 0]
            im_paths += ["/".join(splits[-2::])]

        csv_df[self.image_col] = im_paths

        # check which rows in csv are valid, based on all the channels i want being present
        if self.check_files:
            # TODO add checking to make sure number of keys in h5 file matches number of lines in csv file
            if self.verbose:
                print("Checking the existence of files")

            for index in tqdm(
                range(0, csv_df.shape[0]), desc="Checking files", ascii=True
            ):
                is_good_row = True

                row = csv_df.loc[index]

                image_path = os.sep + row[self.image_col]

                try:
                    self.load_image(row)
                except:  # noqa
                    print("Could not load from image. " + image_path)
                    is_good_row = False

                csv_df.loc[index, "valid_row"] = is_good_row

            # only work with valid rows
            n_old_rows = len(csv_df)
            csv_df = csv_df.loc[csv_df["valid_row"] == True]  # noqa
            csv_df = csv_df.drop("valid_row", 1)
            n_new_rows = len(csv_df)
            if self.verbose:
                print(
                    "{0}/{1} samples have all files present".format(
                        n_new_rows, n_old_rows
                    )
                )

        # Psuedorandomly deterministically convert the cellID to a number for cross validation splits

        rand_split = str2rand(csv_df["CellId"], self.split_seed)
        rand_dna_memb = str2rand(csv_df["CellId"], self.split_seed + 1)

        csv_df["rand_split"] = rand_split
        csv_df["rand_dna_memb"] = rand_dna_memb

        # munge the data so we have an additional DNA and Membrane classes that are randomly sampled from the data here
        image_classes = list(csv_df[self.target_col])
        [label_names, labels] = np.unique(image_classes, return_inverse=True)

        n_classes = len(label_names)
        fract_sample_labels = 1 / n_classes

        duplicate_inds = csv_df["rand_dna_memb"] <= fract_sample_labels

        df_memb = csv_df[duplicate_inds].copy()
        df_memb["ch_struct"] = df_memb["ch_memb"]
        df_memb[self.target_col] = "CellMask"

        df_dna = csv_df[duplicate_inds].copy()
        df_dna["ch_struct"] = df_memb["ch_dna"]
        df_dna[self.target_col] = "Hoechst"

        csv_df = pd.concat([csv_df, df_memb, df_dna]).reset_index()

        image_classes = list(csv_df[self.target_col])
        self.csv_data = csv_df
        self.image_classes = image_classes

        nimgs = len(csv_df)

        [label_names, labels] = np.unique(image_classes, return_inverse=True)
        self.label_names = label_names

        onehot = np.zeros((nimgs, np.max(labels) + 1))
        onehot[np.arange(nimgs), labels] = 1

        self.labels = labels
        self.labels_onehot = onehot

        self.data["test"] = {}
        self.data["test"]["inds"] = np.where(csv_df["rand_split"] <= self.hold_out)[0]

        self.data["train"] = {}
        self.data["train"]["inds"] = np.where(csv_df["rand_split"] > self.hold_out)[0]

        self.imsize = self.load_image(csv_df.iloc[0]).shape

        self.batch_size = batch_size

        self.embeddings = {}
        self.embeddings["train"] = torch.zeros([len(self.data["train"]["inds"]), 0])
        self.embeddings["test"] = torch.zeros([len(self.data["train"]["inds"]), 0])

        self.n_dat = {}

        if n_dat == -1:
            self.n_dat["train"] = len(self.data["train"]["inds"])
        else:
            self.n_dat["train"] = n_dat

        self.n_dat["test"] = len(self.data["test"]["inds"])

    def load_image(self, df_row):

        im_path = self.image_parent + os.sep + df_row[self.image_col]

        if self.return2D:
            im_tmp = np.asarray(Image.open(im_path)).astype("float")
            im_tmp = im_tmp.transpose([2, 0, 1])
        else:
            im_tmp = tifffile.imread(im_path)
            im_tmp = im_tmp.transpose([0, 2, 3, 1])

        ch_names = [self.channel_dict[i] for i in self.channelInds]
        ch_inds = [df_row[ch_name] for ch_name in ch_names]

        im = im_tmp[ch_inds]

        # 3D images are not cropped, so we crop them with the masks
        if not self.return2D:

            for i, ch_name in enumerate(ch_names):
                if ch_name == "ch_dna":  # if dna, crop with dna seg
                    seg_nuc = im_tmp[df_row["ch_seg_nuc"]] > 0

                    im[i] = im[i] * seg_nuc
                elif ch_name == "ch_trans":  # if transmitted, crop dont crop
                    pass  # its transmitted, dont crop
                else:  # otherwise crop with cell
                    seg_cell = im_tmp[df_row["ch_seg_cell"]] > 0

                    im[i] = im[i] * seg_cell

        im = im / 255
        return im

    def set_n_dat(self, n_dat, train_or_test="train"):
        if n_dat == -1:
            self.n_dat[train_or_test] = len(self.data[train_or_test]["inds"])
        else:
            self.n_dat[train_or_test] = n_dat

    def get_n_dat(self, train_or_test="train", override=False):

        if override:
            n_dat = len(self.data[train_or_test]["inds"])
        else:
            n_dat = self.n_dat[train_or_test]

        return n_dat

    def __len__(self, train_or_test="train"):
        return self.get_n_dat(train_or_test)

    def get_image_paths(self, inds_tt, train_or_test):
        inds_master = self.data[train_or_test]["inds"][inds_tt]

        image_paths = list()

        for i, (rownum, row) in enumerate(self.csv_data.iloc[inds_master].iterrows()):
            image_path = self.image_parent + os.sep + row[self.image_col]

            image_paths.append(image_path)

        return image_paths

    def get_images(self, inds_tt, train_or_test):
        dims = list(self.imsize)
        dims[0] = len(self.channelInds)
        dims.insert(0, len(inds_tt))

        inds_master = self.data[train_or_test]["inds"][inds_tt]

        images = torch.zeros(tuple(dims))

        for i, (rownum, row) in enumerate(self.csv_data.iloc[inds_master].iterrows()):
            image = self.load_image(row)

            images[i] = torch.from_numpy(image)

        if self.rescale_to is not None:
            images = torch.nn.functional.interpolate(
                images, scale_factor=self.rescale_to
            )

        if self.crop_to is not None:
            crop = (np.array(images.shape[2:]) - np.array(self.crop_to)) / 2
            crop_pre = np.floor(crop).astype(int)
            crop_post = np.ceil(crop).astype(int)

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

        return images

    def get_classes(self, inds_tt, train_or_test, index_or_onehot="index"):
        inds_master = self.data[train_or_test]["inds"][inds_tt]

        if index_or_onehot == "index":
            labels = self.labels[inds_master]
        else:
            labels = np.zeros([len(inds_master), self.get_n_classes()])
            for c, i in enumerate(inds_master):
                labels[c, :] = self.labels_onehot[i, :]

            labels = torch.from_numpy(labels).long()

        labels = torch.LongTensor(labels)
        return labels

    def get_n_classes(self):
        return self.labels_onehot.shape[1]

    def set_ref(self, embeddings):
        self.embeddings = embeddings

    def get_ref(self, inds, train_or_test="train"):
        inds = torch.LongTensor(inds)
        return self.embeddings[train_or_test][inds]

    def get_n_ref(self):
        return self.get_ref([0], "train").shape[1]

    def get_sample(self, train_or_test="train", inds=None):

        if inds is None:
            rand_inds_encD = np.random.permutation(self.get_n_dat(train_or_test))
            inds = rand_inds_encD[0 : self.batch_size]  # noqa

        x = self.get_images(inds, train_or_test)

        classes = self.get_classes(inds, train_or_test)
        ref = self.get_ref(inds, train_or_test)

        return x, classes, ref
