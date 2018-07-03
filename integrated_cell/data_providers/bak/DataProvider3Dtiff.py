import glob
import os
import numpy as np
from scipy import misc
from natsort import natsorted
from PIL import Image
import torch
import h5py

import pdb

import aicsimage.processing as proc
from aicsimage.io import tifReader



# create a triply nested dict of image paths, with schema
# image_dict[structure_dir][image_name][channel]

def make_image_path_dict(parent_dir):
    
    image_dict = {}
    
    # for each type of structure
    for structure_dir in os.listdir(parent_dir):
        if structure_dir not in image_dict:
            image_dict[structure_dir] = {}
        
        # get all the files in that structure dir
        file_list = os.listdir(os.path.join(parent_dir, structure_dir))
        
        # find unique image prefix strings
        for file_name in file_list:
            image_name, channel_string = file_name.rsplit('_',1)
            channel, tif_ext = os.path.splitext(channel_string)
            if image_name not in image_dict[structure_dir]:
                image_dict[structure_dir][image_name] = {}
            
            # add the paths to all of that image's channels
            image_dict[structure_dir][image_name][channel] = os.path.join(parent_dir,structure_dir,file_name)
    
    return(image_dict)


# load one tiff image, using one list of channel paths (from the list of lists)
def load_tiff(channel_path_list):
    
    # build a dict of numpy arrays, one key/value for each channel
    img = {}
    for i,channel_path in enumerate(channel_path_list):
        with tifReader.TifReader(channel_path) as r:
            # img[channel] is now a greyscale ZYX numpy array
            img[i] = r.load()
            
            
            # transpose Z and Y?
            img[i] = img[i].swapaxes(0,2)

    # convert dict to list
    image = [img[i] for i in natsorted(img.keys())]

    # normalize all channels independently
    for i,_ in enumerate(image):
        image[i] = np.expand_dims(image[i], 0)
        image[i] = image[i] - np.min(image[i])
        image[i] = image[i] / np.max(image[i])

    # turn the list into one big numpy array
    image = np.concatenate(image,0)
    

    return(image)



"""
note for channelInds option:
    0 = memb
    1 = struct
    2 = dna
    3 = cell
    4 = nuc
"""
class DataProvider(object):
    
    def __init__(self, image_parent, opts={}):
        self.data = {}
        
        opts_default = {'rotate': False, 'hold_out': 1/20, 'verbose': False, 'channelInds': [0,1,2]}
                
        # set default values if they are missing
        for key in opts_default.keys(): 
            if key not in opts: 
                opts[key] = opts.get(key, opts_default[key])
        
        self.opts = opts
        
        self.channel_ind_map = {0:'memb', 1:'struct', 2:'dna', 3:'cell', 4:'nuc'}
        self.ind_channel_map = {'memb':0, 'struct':1, 'dna':2, 'cell':3, 'nuc':4}
        self.selected_channel_names = [self.channel_ind_map[i] for i in self.opts['channelInds']]

        # assume file structure is <image_parent>/<class directory>/*.png

        # create big dict of all file paths -- image_dict[structure_dir][image_name][channel]
        image_path_dict = make_image_path_dict(image_parent)
        
        # make image paths: a list of lists:
        # main list is images, sublists are the paths to the desired channels for each image
        image_paths = []
        image_classes = []
        for structure_key in image_path_dict:
            for image_key in image_path_dict[structure_key]:
                list_of_channel_paths = []
                for channel_ind in natsorted(self.channel_ind_map.keys()):
                    list_of_channel_paths += [image_path_dict[structure_key][image_key][self.channel_ind_map[channel_ind]]]
                image_paths += [list_of_channel_paths]
                image_classes = image_classes + [structure_key]
        
        self.image_paths = image_paths
        self.image_classes = image_classes
        
        nimgs = len(image_paths)

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
        
        self.imsize = load_tiff(image_paths[0]).shape
        
    def get_n_dat(self, train_or_test = 'train'):
        return len(self.data[train_or_test]['inds'])
    
#     def get_n_train(self):
#         return len(self.data['train']['inds'])
    
#     def get_n_test(self):
#         return len(self.data['test']['inds'])
    
    def get_n_classes(self):
        return self.labels_onehot.shape[1]
        
    def get_images(self, inds, train_or_test):
        dims = list(self.imsize)
        dims[0] = len(self.opts['channelInds'])
        
        dims.insert(0, len(inds))
        
        images = torch.zeros(tuple(dims))
        
        for c,i in enumerate(inds):
            tif_paths = self.image_paths[self.data[train_or_test]['inds'][i]]
            tif_paths = [tif_paths[i] for i in self.opts['channelInds']]
            image = load_tiff(tif_paths)
            images[c] = torch.from_numpy(image)
        
        # images *= 2
        # images -= 1
        
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
    
    def get_rand_images(self, batsize, train_or_test):
        ndat = self.data[train_or_test]['inds']
        
        inds = np.random.choice(ndat, batsize)
        
        return self.get_images(inds, train_or_test)
    
