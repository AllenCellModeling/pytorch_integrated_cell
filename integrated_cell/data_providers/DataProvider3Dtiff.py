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

import hashlib

import pdb

import aicsimage.processing as proc
from aicsimage.io import tifReader


from integrated_cell.data_providers.DataProviderABC import DataProviderABC 

class DataProvider(DataProviderABC):
    
    def __init__(self, image_parent, batch_size, csv_name='data_jobs_out.csv', opts={}):
        self.data = {}
        
        opts_default = {'rotate': False,
                        'hold_out': 1/10,
                        'verbose': True,
                        'target_col':'structureProteinName',
                        'channelInds': [0, 1, 2],
                        'h5_file': True,
                        'check_files':True,
                        'split_seed': 1}
                
        # set default values if they are missing
        for key in opts_default.keys(): 
            if key not in opts: 
                opts[key] = opts.get(key, opts_default[key])
        
        self.opts = opts
        self.image_parent = image_parent
        self.csv_name = csv_name

        # translate short channel name spassed in into longer csv column names
        self.channel_index_to_column_dict = {
            0:'save_cell_reg_path',
            1:'save_nuc_reg_path',
            2:'save_dna_reg_path',
            3:'save_memb_reg_path',
            4:'save_struct_reg_path',
            5:'save_trans_reg_path'
        }
        
        self.channel_lookup_table = np.asarray([
            3, #0 means memb
            4, #1 means struct
            2, #2 means dna
            0, #3 means seg cell
            1, #4 means seg nuc
            5  #5 means transmitted
        ])
        
        channel_column_list = [col for col in self.channel_index_to_column_dict.values()]
        
        # make a dataframe out of the csv log file
        csv_path = os.path.join(self.image_parent,self.csv_name)
        if self.opts['verbose']:
            print('reading csv manifest')
        csv_df = pd.read_csv(csv_path)
        
        # check which rows in csv are valid, based on all the channels i want being present
        if self.opts['check_files'] is not None:
                    # TODO add checking to make sure number of keys in h5 file matches number of lines in csv file
            for index in tqdm(range(0, csv_df.shape[0]), desc='Checking files', ascii=True):
                is_good_row = True
     
                row = csv_df.loc[index]

                try:
                    image_paths = image_parent + os.sep + row[channel_column_list]
                    d = self.load_tiff(image_paths)
                except:
                    print('Could not load from image. ' + image_paths[0])
                    is_good_row = False
                
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
        
        hash_strings = csv_df.save_h5_reg_path
        
        #we're gonna psuedorandomly deterministically convert the path to the file to a number for cross validation
        
        #prepend the 'seed' to the unique file path, then hash with SHA512
        salted_string = str(opts['split_seed']) + csv_df.save_h5_reg_path
        hash_strings = [hashlib.sha512(string.encode('utf-8')).hexdigest() for string in salted_string]
            
        img_nums = list()
        
        #Pull out the first 5 digits to get a value between 0-1 inclusive
        for hash_string in hash_strings:
            str_nums = [char for pos, char in enumerate(hash_string) if char.isdigit()]
            str_num = ''.join(str_nums[0:5])
            num = float(str_num)/100000
            img_nums.append(num)
        
        self.data['test'] = {}
        self.data['test']['inds'] = np.where(np.array(img_nums) <= opts['hold_out'])[0]
        
        self.data['train'] = {}
        self.data['train']['inds'] = np.where(np.array(img_nums) > opts['hold_out'])[0]
        
        self.imsize = self.load_tiff(image_paths).shape
        
        self.ndat['train'] = len(self.data['train']['inds'])
        self.ndat['test'] = len(self.data['test']['inds'])
        
        self.batch_size = batch_size
        
        self.embeddings['train'] = torch.zeros(len(self.data['train']['inds']))
        self.embeddings['test'] = torch.zeros(len(self.data['train']['inds']))
    
    # load one tiff image, using one row index of the (potentially filtered) csv dataframe
    def load_tiff(self, channel_paths):

        # build a list of numpy arrays, one for each channel
        image = list()

        for channel_path in channel_paths: 
            with tifReader.TifReader(channel_path) as r:
                channel = r.load() # ZYX numpy array


                channel = channel.transpose(0,2,1,3,4) # transpose Z and Y -> XYZ
                # channel = np.expand_dims(channel, axis=0)
                image.append(channel)

        # turn the list into one big numpy array
        image = np.concatenate(image,1)
        image = np.squeeze(image)

        return image
    
    def set_n_dat(self, n_dat, train_or_test = 'train'):
        self.n_dat[train_or_test] = n_dat
        
    def get_n_dat(self, train_or_test = 'train'):
        return self.n_dat[train_or_test]
    
    
    def __len__(self):
        return get_n_dat(train_or_test = 'train')
    
    def get_n_classes(self):
        return self.labels_onehot.shape[1]
        
    def get_image_paths(self, inds_tt, train_or_test):
        inds_master = self.data[train_or_test]['inds'][inds_tt]
        
        image_paths = list()
        
        for i, (rownum, row) in enumerate(self.csv_data.iloc[inds_master].iterrows()):
            h5_file = self.image_parent + os.sep +  row.save_h5_reg_path

            image_paths.append(h5_file)
        
        return image_paths
        
    def get_images(self, inds_tt, train_or_test):
        dims = list(self.imsize)
        
        dims[0] = len(self.opts['channelInds'])
        dims.insert(0, len(inds_tt))
        
        inds_master = self.data[train_or_test]['inds'][inds_tt]
        
        images = torch.zeros(tuple(dims))
        
        for i, (rownum, row) in enumerate(self.csv_data.iloc[inds_master].iterrows()):
            h5_file = self.image_parent + os.sep +  row.save_h5_reg_path
            image = self.load_h5(h5_file)
            
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
            
            labels = torch.from_numpy(labels).long()
            
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

    def set_ref(self, embeddings):
        self.embeddings = embeddings
    
    def get_ref(self, inds, train_or_test='train'):
        inds = torch.LongTensor(inds)
        return self.embeddings[train_or_test][inds]
    
    def get_sample(self, train_or_test = 'train'):
        
        rand_inds_encD = np.random.permutation(self.get_n_dat(train_or_test))
        inds = rand_inds_encD[0:self.batch_size]
        
        x = self.get_images(inds, train_or_test)
        classes = self.get_classes(inds, train_or_test)
        ref = self.ref(inds, train_or_test)