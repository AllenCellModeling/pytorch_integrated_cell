from pathlib import Path
import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
import pandas as pd
import glob

from tqdm import tqdm
from PIL import Image
from aicsimageio import AICSImage

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from ..utils.utils import str2rand
from integrated_cell.data_providers.DataProviderABC import DataProviderABC

def cellpath2dict(path):
    cell = path.split("/")[-1]
    cell = cell.split(".")[0]
    cell = cell.split("_")
    return {
        cell[i*2]: cell[i*2 + 1]
        for i in range(len(cell)//2)
    }

def make_manifest(dataset_path="/allen/aics/modeling/gui.pires/dev/scratchpad/aics_mnist/aics_mnist_rgb"):
    cells = []
    for split in ["train", "test"]:
        _split_str = str((Path(dataset_path) / split) / "*")
        for structure_path in glob.glob(_split_str):
            _path_str = str(Path(structure_path) / "*")
            structure = structure_path.split("/")[-1]
            for cell_img in glob.glob(_path_str):
                cells.append(
                    dict(cellpath2dict(cell_img), structure=structure, split=split,
                         path=str(Path(cell_img).resolve()))
                )

    return pd.DataFrame(cells)

def make_tensor_dataset(df, channel_ixs, verbose=True):
    if verbose:
        print("loading dataset into GPU")
        rows = tqdm(df.iterrows(), total=len(df))
    else:
        rows = df.iterrows()

    return TensorDataset(
        torch.stack(
            [TF.to_tensor(Image.open(row.path))[channel_ixs]
             for _, row in rows]
        ).cuda()
    )

class DataProvider(DataProviderABC):
    def __init__(
        self,
        image_parent,
        batch_size,
        n_dat=-1,
        hold_out=0.2,
        dataset_folder="/allen/aics/modeling/gui.pires/scratchpad/aics_mnist/aics_mnist_rgb",
        channel_names=["membrane", "dna"],
        check_files=True,
        split_seed=1,
        rescale_to=None,
        crop_to=None,
        make_controls=False,
        normalize_intensity=False,
        verbose=True,
    ):

        self.data = {}

        self.hold_out = hold_out
        self.verbose = verbose
        self.channel_names = channel_names
        self.check_files = check_files
        self.split_seed = split_seed
        self.crop_to = crop_to
        self.rescale_to = rescale_to
        self.image_parent = image_parent
        self.normalize_intensity = normalize_intensity
        self.channel_dict = {
            'membrane': 0,
            'structure': 1,
            'dna': 2,
        }

        csv_df = make_manifest(dataset_folder)
        self.csv_data = csv_df
        self.image_classes = list(csv_df["structure"])
        self.tensor_dataset = make_tensor_dataset(csv_df, [self.channel_dict[ch] for ch in channel_names], verbose=True)

        nimgs = len(self.csv_data)

        [label_names, labels] = np.unique(self.image_classes, return_inverse=True)
        self.label_names = label_names

        onehot = np.zeros((nimgs, np.max(labels) + 1))
        onehot[np.arange(nimgs), labels] = 1

        self.labels = labels
        self.labels_onehot = onehot

        self.data["test"] = {}
        self.data["test"]["inds"] = np.where(csv_df["split"] == "test")[0]

        self.data["validate"] = {}
        self.data["train"] = {}

        train_inds, val_inds = train_test_split(
            csv_df.loc[csv_df["split"] == "train"].index,
            test_size=self.hold_out,
            stratify=csv_df.loc[csv_df["split"] == "train"].structure,
            random_state=split_seed,
        )

        self.data["train"]["inds"] = train_inds
        self.data["validate"]["inds"] = val_inds

        self.imsize = (28, 28)

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
        # in pandas, a row's name contains its index
        return self.tensor_dataset[df_row.name][0]

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
        inds_master = self.data[train_or_test]["inds"][inds_tt]

        return self.tensor_dataset[inds_master][0]

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

        #return x, classes, ref
        return x
