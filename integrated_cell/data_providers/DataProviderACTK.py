from pathlib import Path
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from aicsimageio import AICSImage

from ..utils.utils import str2rand
from integrated_cell.data_providers.DataProviderABC import DataProviderABC


class DataProvider(DataProviderABC):
    def __init__(
        self,
        image_parent,
        batch_size,
        n_dat=-1,
        csv_name="manifest.csv",
        hold_out=0.2,
        verbose=True,
        image_col="CellImage3DPath",
        target_col="ProteinDisplayName",
        channel_names=["membrane_segmentation", "structure", "nucleus_segmentation"],
        check_files=True,
        split_seed=1,
        return2D=False,
        rescale_to=None,
        crop_to=None,
        make_controls=False,
        normalize_intensity=False,
    ):

        self.data = {}

        self.hold_out = hold_out
        self.verbose = verbose
        self.image_col = image_col
        self.target_col = target_col
        self.channel_names = channel_names
        self.check_files = check_files
        self.split_seed = split_seed
        self.crop_to = crop_to
        self.rescale_to = rescale_to
        self.image_parent = image_parent
        self.csv_name = csv_name
        self.normalize_intensity = normalize_intensity
        self.return2D = return2D
        self.channel_dict = {
            'nucleus_segmentation': 0,
            'membrane_segmentation': 1,
            'dna': 2,
            'membrane': 3,
            'structure': 4,
            'brightfield': 5,
        }

        # make a dataframe out of the csv log file
        csv_path = Path(self.image_parent) / Path(self.csv_name)
        if self.verbose:
            print("reading csv manifest")
        csv_df = pd.read_csv(csv_path)

        # check which rows in csv are valid, based on all the channels i want being present
        if self.check_files:
            if self.verbose:
                print("Checking the existence of files")
                
            csv_df["valid_row"] = False
            for index,row in tqdm(csv_df.iterrows(), total=len(csv_df), desc="Checking files"):
                try:
                    self.load_image(row)
                    is_good_row = True
                except:  # noqa
                    print(f"Could not load from image. {row[self.image_col]}")
                    is_good_row = False
                csv_df.at[index, "valid_row"] = is_good_row

            # only work with valid rows
            n_old_rows = len(csv_df)
            csv_df = csv_df.loc[csv_df["valid_row"] == True]  # noqa
            csv_df = csv_df.drop("valid_row", 1)
            n_new_rows = len(csv_df)
            if self.verbose:
                print(f"{n_new_rows}/{n_old_rows} samples have all files present")

        # Psuedorandomly deterministically convert the cellID to a number for cross validation splits
        rand_split = str2rand(csv_df["CellId"], self.split_seed)
        rand_dna_memb = str2rand(csv_df["CellId"], self.split_seed + 1)

        csv_df["rand_split"] = rand_split
        csv_df["rand_dna_memb"] = rand_dna_memb

        csv_df = csv_df.reset_index()

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

        self.data["validate"] = {}
        self.data["validate"]["inds"] = np.where(
            (csv_df["rand_split"] > self.hold_out)
            & (csv_df["rand_split"] <= self.hold_out * 2)
        )[0]

        self.data["train"] = {}
        self.data["train"]["inds"] = np.where(csv_df["rand_split"] > self.hold_out * 2)[
            0
        ]

        self.imsize = self.load_image(csv_df.iloc[0]).shape

        self.batch_size = batch_size

        self.embeddings = {}
        self.embeddings["train"] = torch.zeros([len(self.data["train"]["inds"]), 0])
        self.embeddings["test"] = torch.zeros([len(self.data["test"]["inds"]), 0])
        self.embeddings["validate"] = torch.zeros(
            [len(self.data["validate"]["inds"]), 0]
        )

        self.n_dat = {}

        if n_dat == -1:
            self.n_dat["train"] = len(self.data["train"]["inds"])
        else:
            self.n_dat["train"] = n_dat

        self.n_dat["validate"] = len(self.data["validate"]["inds"])
        self.n_dat["test"] = len(self.data["test"]["inds"])

    def load_image(self, df_row):

        im_tmp = AICSImage(df_row[self.image_col])
        im_tmp = im_tmp.get_image_data("CYXZ")
        
        ch_inds = [self.channel_dict[ch_name] for ch_name in self.channel_names]
        im = im_tmp[ch_inds]

        # 3D images are not cropped to masks, so we mask them
        for i, ch_name in enumerate(self.channel_names):
            seg_channel = "nucleus_segmentation" if ch_name == "dna" else "membrane_segmentation"
            seg = im_tmp[self.channel_dict[seg_channel]] > 0
            im[i] = im[i] * seg

        if self.normalize_intensity:
            im = im.astype("float")

            if str(self.normalize_intensity).lower() == "max":
                for i, ch in enumerate(im):
                    if np.max(im[i]) > 0:
                        im[i] = ch / np.max(ch)
            
            if str(self.normalize_intensity).lower() == "std":
                for i, ch in enumerate(im):
                    if np.sum(im[i]) > 0:
                        im[i] = ch / np.std(ch)

            if str(self.normalize_intensity).lower() == "avg_intensity":
                # normalize each channel to the average intensity of each test set channel
                for i, ch in enumerate(im):
                    if np.sum(im[i]) > 0:
                        ch = ch / np.sum(ch)
                    else:
                        ch = 1 / np.size(ch)

                    im[i] = ch * 618.0294

            elif self.normalize_intensity == 1:
                for i, ch in enumerate(im):
                    if np.sum(im[i]) > 0:
                        im[i] = ch / np.sum(ch)
                    else:
                        im[i] = 1 / np.size(ch)

        else:
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
            image_paths.append(row[self.image_col])
        return image_paths

    
    def get_images(self, inds_tt, train_or_test):
        dims = list(self.imsize)
        dims[0] = len(self.channel_names)
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
