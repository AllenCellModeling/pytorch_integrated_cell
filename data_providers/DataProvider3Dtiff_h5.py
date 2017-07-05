# from commit e57a711, https://github.com/AllenCellModeling/pytorch_segmentation_classifier

import glob
import os
import numpy as np
from scipy import misc
from natsort import natsorted
from PIL import Image
import torch
import h5py
import pandas as pd
import copy
import random
from tqdm import tqdm

import pdb

import aicsimage.processing as proc
from aicsimage.io import tifReader


# load one tiff image, using one row index of the (potentially filtered) csv dataframe
def load_tiff(df_row, channel_cols=['save_memb_reg_path', 'save_struct_reg_path', 'save_dna_reg_path'], resize=None):
    
    # build a list of numpy arrays, one for each channel
    image = []
    for channel_name in channel_cols:
        channel_path = df_row[channel_name]
        channel_path = channel_path.split('/allen/aics/')[1]
        channel_path = os.path.join('/root',channel_path)
        
        with tifReader.TifReader(channel_path) as r:
            channel = r.load()               # ZYX numpy array
            channel = channel.swapaxes(0,2)  # transpose Z and Y -> XYZ
            if resize is not None:
                channel = proc.resize(channel, resize, "bilinear")
            image += [channel]
            
    # normalize all channels independently
    for i,_ in enumerate(image):
        image[i] = np.expand_dims(image[i], 0)
        image[i] = image[i] - np.min(image[i])
        image[i] = image[i] / np.max(image[i])

    # turn the list into one big numpy array
    image = np.concatenate(image,0)

    return(image)


class DataProvider(object):
    
    def __init__(self, image_parent, csv_name='data_jobs_out.csv', opts={}):
        self.data = {}
        
        opts_default = {'rotate': False,
                        'hold_out': 1/20,
                        'verbose': False,
                        'target_col':'FinalScore',
                        'channel_names': ['memb', 'struct', 'dna'],
                        'h5_file':'all_images.h5',
                        'resize':None,
                        'preload':False}
                
        # set default values if they are missing
        for key in opts_default.keys(): 
            if key not in opts: 
                opts[key] = opts.get(key, opts_default[key])
        
        self.opts = opts
        self.image_parent = image_parent
        self.csv_name = csv_name

        # translate short channel name spassed in into longer csv column names
        channel_name_to_column_dict = {
            'cell':'save_cell_reg_path',
            'nuc':'save_nuc_reg_path',
            'dna':'save_dna_reg_path',
            'memb':'save_memb_reg_path',
            'struct':'save_struct_reg_path'
        }
        
        index_to_channel_name_to_dict = {
            0:'cell',
            1:'nuc',
            2:'dna',
            3:'memb',
            4:'struct'
        }
        
        # if using channels numbers as names (eww) then translate the dict keys:
        channel_cols = []
        for c in self.opts['channel_names']:
            if isinstance(c, int):
                k = index_to_channel_name_to_dict[c]
            else:
                k = c
            channel_cols += [channel_name_to_column_dict[k]]
        
        self.channel_cols = channel_cols
        
        
        if self.opts['verbose']:
            print('reading from columns: {}'.format(self.channel_cols))
        
        # make a dataframe out of the csv log file
        csv_path = os.path.join(self.image_parent,self.csv_name)
        if self.opts['verbose']:
            print('reading csv manifest')
        csv_df = pd.read_csv(csv_path)
        
        
        # check which rows in csv are valid, based on all the channels i want being present
        if self.opts['check_files']:
            if self.opts['verbose']:
                print('checking to see if all files for each image are present')

            csv_df['valid_row'] = True
            for index, row in csv_df.iterrows():
                is_good_row = True
                for channel in tqdm(channel_cols, desc='checking rows in csv', ascii=True):
                    channel_path = row[channel]
                    # TODO remove path hack when greg fixes abs -> rel paths
                    channel_path = channel_path.split('/allen/aics/')[1]
                    channel_path = os.path.join('/root',channel_path)
                    is_good_row = is_good_row and os.path.exists(channel_path)
                csv_df.loc[index, 'valid_row'] = is_good_row


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
        
        self.imsize = load_tiff(csv_df.iloc[0], channel_cols=self.channel_cols, resize=self.opts['resize']).shape
        
        # if the h5_file option is set, load the files in to an h5 file if it doesn't exist
        h5f_path = os.path.join(self.image_parent,self.opts['h5_file'])
        if not os.path.exists(h5f_path):
            # TODO add checking to make sure number of keys in h5 file matches number of lines in csv file
            if self.opts['verbose']:
                print('need to make h5 file')
            h5f = h5py.File(h5f_path, 'w')

            for (i, row) in tqdm(self.csv_data.iterrows(), desc='loading images into h5 file', ascii=True):
                d = load_tiff(row, channel_cols=self.channel_cols, resize=self.opts['resize'])
                # h5 keys are the row numbers in the csv file
                h5f.create_dataset(str(i), data=d)
            h5f.close()
            
        # if the preload option is set, load the images into one giant numpy array in main memory
        if self.opts['preload']:
            if self.opts['h5_file']:                
                h5f_path = os.path.join(self.image_parent,self.opts['h5_file'])
                h5f = h5py.File(h5f_path,'r')
                self.images_preloaded = np.zeros([len(h5f.keys()), *self.imsize])
                for (i, row) in tqdm(self.csv_data.iterrows(), desc='loading images into memory', ascii=True):
                    image = h5f[str(i)][:]
                    self.images_preloaded[i,:,:,:,:] = image
                h5f.close()
            else:
                raise ValueError('preload = True only supported if h5_file is not None')
            
            # persist preloaded array as part of the dataprovider
            # self.images_preloaded = images_preloaded_np_array
        
    def get_n_dat(self, train_or_test = 'train'):
        return len(self.data[train_or_test]['inds'])
    
    def get_n_classes(self):
        return self.labels_onehot.shape[1]
        
    def get_images(self, inds_tt, train_or_test):
        dims = list(self.imsize)
        dims[0] = len(self.opts['channel_names'])
        dims.insert(0, len(inds_tt))
        
        inds_master = self.data[train_or_test]['inds'][inds_tt]
        
        images = torch.zeros(tuple(dims))
        
        # use the h5 file if we have it or made it -- much faster io than reading tiffs
        if self.opts['h5_file'] is not None:
            
            # use preloaded images if we have them
            if self.opts['preload'] is not None :
                for i,k in enumerate(inds_master):
                    images[i] = torch.from_numpy(self.images_preloaded[k])
                # images = self.images_preloaded[inds_master]
            
            # otherwise load on demand
            else:
                h5f_path = os.path.join(self.image_parent,self.opts['h5_file'])
                h5f = h5py.File(h5f_path,'r')
                for i,k in enumerate(inds_master):
                    image = h5f[str(k)][:]
                    images[i] = torch.from_numpy(image)
                h5f.close()
                
        # use tiff reader if no better option
        else:
            for i, (rownum, row) in enumerate(self.csv_data.iloc[inds_master].iterrows()):
                image = load_tiff(row, channel_cols=self.channel_cols,resize=self.opts['resize'])
                images[i] = torch.from_numpy(image)
        
        return images
    
    def get_classes(self, inds_tt, train_or_test, index_or_onehot = 'index'):
        
        inds_master = self.data[train_or_test]['inds'][inds_tt]

        if index_or_onehot == 'index':
            labels = self.labels[inds_master]
        else:
            labels = np.zeros([len(inds_master), self.get_n_classes()])
            for c,i in enumerate(inds_master):
                labels[c,:] = self.labels_onehot[i,:]
        
        labels = torch.LongTensor(labels)
        return labels
    
    # train minibatches cycle through entire data set
    # test minibatches are random draws each time to match the size of train minibatch
    def make_random_minibatch_inds_train_and_test(self, batch_size=8):
        
        n_train = self.get_n_dat('train')
        inds_train_shuf = random.sample(range(n_train),n_train)
        mini_batches_inds_train = [inds_train_shuf[i:i + batch_size] for i in range(0, len(inds_train_shuf), batch_size)]
        
        mini_batch_train_lens = [len(b) for b in mini_batches_inds_train]
        n_test = self.get_n_dat('test')
        mini_batches_inds_test = [random.sample(range(n_test),b_size) for b_size in mini_batch_train_lens]
        
        minibatch_inds_list = {}
        minibatch_inds_list['train'] = mini_batches_inds_train
        minibatch_inds_list['test'] = mini_batches_inds_test
        
        return(minibatch_inds_list)