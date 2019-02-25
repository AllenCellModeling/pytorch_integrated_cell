import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import tifffile
import hashlib

from integrated_cell.data_providers.DataProviderABC import DataProviderABC


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
        image_col="save_reg_path",
        channelInds=np.arange(3, 15),
        check_files=True,
        split_seed=1,
        crop_to=None,
        return2D=False,
        channel_lookup=np.arange(0, 15),
    ):

        self.data = {}

        self.hold_out = hold_out
        self.verbose = verbose
        self.target_col = target_col
        self.image_col = image_col
        self.channelInds = channelInds
        self.check_files = check_files
        self.split_seed = split_seed
        self.crop_to = crop_to

        self.image_parent = image_parent
        self.csv_name = csv_name

        self.return2D = return2D

        # translate short channel name spassed in into longer csv column names

        self.channel_lookup = channel_lookup

        # channel_column_list = [col for col in self.channel_index_to_column_dict.values()]

        # make a dataframe out of the csv log file
        csv_path = os.path.join(self.image_parent, self.csv_name)
        if self.verbose:
            print("reading csv manifest")
        csv_df = pd.read_csv(csv_path)

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
                    tifffile.imread(image_parent + os.sep + image_path)
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

        hash_strings = csv_df["CellId"]

        # we're gonna psuedorandomly deterministically convert the path to the file to a number for cross validation

        # prepend the 'seed' to the unique file path, then hash with SHA512
        salted_string = [
            str(self.split_seed) + str(cellid) for cellid in csv_df["CellId"]
        ]
        hash_strings = [
            hashlib.sha512(string.encode("utf-8")).hexdigest()
            for string in salted_string
        ]

        img_nums = list()

        # Pull out the first 5 digits to get a value between 0-1 inclusive
        for hash_string in hash_strings:
            str_nums = [char for pos, char in enumerate(hash_string) if char.isdigit()]
            str_num = "".join(str_nums[0:5])
            num = float(str_num) / 100000
            img_nums.append(num)

        self.data["test"] = {}
        self.data["test"]["inds"] = np.where(np.array(img_nums) <= self.hold_out)[0]

        self.data["train"] = {}
        self.data["train"]["inds"] = np.where(np.array(img_nums) > self.hold_out)[0]

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

    def load_image(self, im_path):
        im_tmp = tifffile.imread(im_path)

        ch_inds = np.array(self.channelInds)
        ch_lookup = np.array(self.channel_lookup)

        im_inds = ch_lookup[ch_inds]

        im_mask = im_tmp[1] > 0

        image = im_tmp[im_inds]

        for i, ch_ind in enumerate(ch_inds):

            if ch_ind == 0 or ch_ind == 1 or ch_ind == 2:
                pass  # skip cropping the segs and the transmitted light
            else:  # otherwise crop with cell
                image[i] = image[i] * im_mask

        image = image.transpose([0, 2, 3, 1])

        if np.any(image >= 255):
            image = image / 255

        if self.crop_to is not None:
            delta = (np.array(image.shape[1:]) - np.array(self.crop_to)) / 2

            crop = np.zeros(delta.shape)
            crop[delta > 0] = delta[delta > 0]
            crop_pre = np.floor(crop).astype(int)
            crop_post = np.ceil(crop).astype(int)

            crop_post[crop_post == 0] = -np.array(image.shape[1:])[crop_post == 0]

            image = image[
                :,
                crop_pre[0] : -crop_post[0],  # noqa
                crop_pre[1] : -crop_post[1],  # noqa
                crop_pre[2] : -crop_post[2],  # noqa
            ]

            pad = np.zeros(delta.shape)
            pad[delta < 0] = -delta[delta < 0]
            pad_pre = np.concatenate([[0], np.floor(pad)]).astype(int)
            pad_post = np.concatenate([[0], np.ceil(pad)]).astype(int)

            pad = [[pre, post] for pre, post in zip(pad_pre, pad_post)]

            image = np.pad(image, pad, mode="constant", constant_values=0)

        return image

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

        inds_master = self.data[train_or_test]["inds"][inds_tt]

        images = []
        for i, (rownum, row) in enumerate(self.csv_data.iloc[inds_master].iterrows()):
            image_path = self.image_parent + os.sep + row[self.image_col]
            image = self.load_image(image_path)

            images.append(image)

        images = np.stack(images, 0)
        images = torch.from_numpy(images).float()

        if self.return2D:
            images = torch.max(images, dim=4)[0]

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

    def __getitem__(self, ind, train_or_test="train"):
        inds_master = self.data[train_or_test]["inds"][ind]

        row = self.csv_data.iloc[inds_master]
        image_path = self.image_parent + os.sep + row[self.image_col]
        image = self.load_image(image_path)

        label = self.labels[inds_master]

        return image, label
