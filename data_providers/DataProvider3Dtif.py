import glob
import os
import numpy as np
from scipy import misc
from natsort import natsorted
from PIL import Image
import torch
import h5py
import pandas as pd
import numpy as np

import pdb

import aicsimage.processing as proc
from aicsimage.io import tifReader


# load one tiff image, using one row index of the (potentially filtered) csv dataframe
def load_tiff(channel_paths):
    
    # build a list of numpy arrays, one for each channel
    image = list()

    for channel_path in channel_paths: 
        with tifReader.TifReader(channel_path) as r:
            channel = r.load() # ZYX numpy array
            channel = channel.transpose(1,2,0) # transpose Z and Y -> XYZ
            channel = np.expand_dims(channel, axis=0)
            image.append(channel)

    # turn the list into one big numpy array
    image = np.concatenate(image,0)

    return(image)


class DataProvider(object):
    
    def __init__(self, image_parent, csv_name='data_jobs_out.csv', opts={}):
        self.data = {}
        
        opts_default = {'rotate': False,
                        'hold_out': 1/20,
                        'verbose': True,
                        'target_col':'structureProteinName',
                        'channel_names': [0, 1, 2],
                        'preload':False,
                        'check_files':True}
                
        # set default values if they are missing
        for key in opts_default.keys(): 
            if key not in opts: 
                opts[key] = opts.get(key, opts_default[key])
        
        self.opts = opts

        # translate short channel name spassed in into longer csv column names
        channel_inds_to_column_dict = {
            0:'save_memb_reg_path',
            1:'save_struct_reg_path',
            2:'save_dna_reg_path',
            3:'save_nuc_reg_path',
            4:'save_cell_reg_path'
        }
        
        
        channel_cols = [channel_inds_to_column_dict[channel_name] for channel_name in self.opts['channel_names']]
        self.channel_cols = channel_cols
        if self.opts['verbose']:
            print('reading from columns: {}'.format(self.channel_cols))
        
        # make a dataframe out of the csv log file
        csv_path = os.path.join(image_parent,csv_name)
        if self.opts['verbose']:
            print('reading csv manifest')
        csv_df = pd.read_csv(csv_path)
        csv_df = csv_df[csv_df['FinalScore'] == 1]
        
        all_channels = [channel_inds_to_column_dict[k] for k in channel_inds_to_column_dict]
        csv_df[all_channels] = image_parent + os.sep + csv_df[all_channels]
        csv_df = csv_df.reset_index()
        
        # check which rows in csv are valid, based on all the channels i want being present
        if self.opts['check_files']:
            if self.opts['verbose']:
                print('checking to see if all files for each image are present')

            csv_df['valid_row'] = True
            for index, row in csv_df.iterrows():
                is_good_row = True
                for channel in channel_cols:
                    channel_path = row[channel]
                    # TODO remove path hack when greg fixes abs -> rel paths
                    # channel_path = channel_path.split('/allen/aics/')[1]
                    # channel_path = os.path.join('/root',channel_path)

                    is_good_row = is_good_row and os.path.exists(channel_path)
                csv_df.loc[index, 'valid_row'] = is_good_row
                if self.opts['verbose']:
                    if index % 250 == 0:
                        print('{0}/{1} images checked'.format(index,len(csv_df)))

            # only work with valid rows
            n_old_rows = len(csv_df)
            csv_df = csv_df.loc[csv_df['valid_row'] == True]
            csv_df = csv_df.drop('valid_row', 1)
            n_new_rows = len(csv_df)
            if self.opts['verbose']:
                print('{0}/{1} samples have all files present'.format(n_new_rows,n_old_rows))
        
        image_classes = list(csv_df[self.opts['target_col']])
        self.csv_data = csv_df
        self.image_classes = image_classes
        
        nimgs = len(csv_df)

        [label_names, labels] = np.unique(image_classes, return_inverse=True)
        self.label_names = label_names
        
        onehot = np.zeros((nimgs, np.max(labels)+1))
        onehot[np.arange(nimgs), labels] = 1
        
        self.labels = labels
        self.labels_onehot = onehot
    
        rand_inds = np.random.permutation(nimgs)
        
        ntest = int(np.round(nimgs*opts['hold_out']))
        
        self.data['test'] = {}
        self.data['test']['inds'] = rand_inds[0:ntest+1]
        
        self.data['train'] = {}
        self.data['train']['inds'] = rand_inds[ntest+2:-1]
        
        self.imsize = load_tiff(csv_df.iloc[0][self.channel_cols]).shape
        
        # if the preload option is set, load all the images into a numpy array
        if self.opts['preload']:
            image_sizes = []
            if self.opts['verbose']:
                print('preloading images into main memory')
            preloaded_image_list = []
            for i, (row_ind, row) in enumerate(self.csv_data.iterrows()):
                image = load_tiff(row, channel_cols=self.channel_cols)
                preloaded_image_list += [image]
                image_sizes += [image.shape]
                if self.opts['verbose']:
                    if i % 100 == 0:
                        print('{0}/{1} images loaded'.format(i,n_new_rows))
            self.image_sizes = image_sizes
            self.preloaded_image_list = preloaded_image_list
            
        
    def get_n_dat(self, train_or_test = 'train'):
        return len(self.data[train_or_test]['inds'])
    
    def get_n_classes(self):
        return self.labels_onehot.shape[1]
        
    def get_images(self, inds, train_or_test):
        dims = list(self.imsize)
        dims.insert(0, len(inds))
        
        dims[0] = len(inds)
        
        
        images = torch.zeros(tuple(dims))
        
        inds = self.data[train_or_test]['inds'][inds]
        
        if self.opts['preload']:
            for i in inds:
                image = self.preloaded_image_list[i]
                images[i] = torch.from_numpy(image)
        else:
            for i, (rownum, row) in enumerate(self.csv_data.iloc[inds].iterrows()):
                image = load_tiff(row[self.channel_cols])
                images[i] = torch.from_numpy(image)
        
        return images
    
    def get_classes(self, inds, train_or_test, index_or_onehot = 'index'):
        
        if index_or_onehot == 'index':
            labels = self.labels[self.data[train_or_test]['inds'][inds]]
        else:
            labels = np.zeros([len(inds), self.get_n_classes()])
            c = 0
            for i in inds:
                labels[c] = self.labels_onehot[self.data[train_or_test]['inds'][i]]
                c += 1
        
        labels = torch.LongTensor(labels)
        return labels
    