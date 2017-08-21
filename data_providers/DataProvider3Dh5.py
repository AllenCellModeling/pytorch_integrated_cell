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

class DataProvider(object):
    
    def __init__(self, image_parent, csv_name='data_jobs_out.csv', opts={}):
        self.data = {}
        
        opts_default = {'rotate': False,
                        'hold_out': 1/10,
                        'verbose': True,
                        'target_col':'structureProteinName',
                        'channelInds': [0, 1, 2],
                        'h5_file': True,
                        'resize':0.795,
                        'pad_to':(128,96,64),
                        'preload':False,
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
            if self.opts['verbose']:
                print('Making h5 files')

            for index in tqdm(range(0, csv_df.shape[0]), desc='loading images into h5 files', ascii=True):
                is_good_row = True

                row = csv_df.loc[index]

                h5_path = self.image_parent + os.sep + row.save_h5_reg_path
                if not os.path.exists(h5_path):
                    
                    try:
                        h5f = h5py.File(h5_path, 'w')
                        image_paths = image_parent + os.sep + row[channel_column_list]
                        d = self.load_tiff(image_paths)
                        # h5 keys are the row numbers in the csv file
                        h5f.create_dataset('image', data=d)
                        h5f.close()
                    except:
                        print('Could not load from image. ' + image_paths[0])
                        if os.path.exists(h5_path):
                            os.remove(h5_path)
                            
                        is_good_row = False
                else:
                    try:
                        self.load_h5(h5_path)
                    except:
                        # pdb.set_trace()
                        print('Could not load ' + h5_path + '. Skipping.')
                        # os.remove(h5_path)
                        
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
        
        self.imsize = self.load_h5(self.image_parent + os.sep +  csv_df.iloc[0].save_h5_reg_path).shape
    
    def load_h5(self, h5_path):
        chInds = self.channel_lookup_table[np.asarray(self.opts['channelInds'])]
        
        f = h5py.File(h5_path,'r')
        
        image = f['image'].value[chInds]
        image = image.astype('double')/255

        return image
    
    def get_n_dat(self, train_or_test = 'train'):
        return len(self.data[train_or_test]['inds'])
    
    def get_n_classes(self):
        return self.labels_onehot.shape[1]
        
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
