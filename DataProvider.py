import glob
import os
import numpy as np
from scipy import misc
from natsort import natsorted
from PIL import Image
import torch


import pdb

class DataProvider(object):
    
    def __init__(self, image_parent, opts={}):
        self.data = {}
        
        opts_default = {'rotate': False, 'hold_out': 1/20, 'out_size': -1, 'verbose': False, 'pattern': '*.png', 'channelInds': [0,1,2]}
        
        # set default values if they are missing
        for key in opts_default.keys(): 
            if key not in opts: 
                opts[key] = opts.get(key, opts_default[key])
        
        self.opts = opts

        # assume file structure is <image_parent>/<class directory>/*.png
        image_dirs = natsorted(glob.glob(image_parent + os.sep + '*'))

        image_paths = list()
        for image_dir in image_dirs:
            image_paths += natsorted(glob.glob(image_dir + os.sep + opts['pattern']))

        image_classes = [os.path.basename(os.path.dirname(image_path)) for image_path in image_paths]

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
        
        if opts['verbose']:
            print('Preloading data')
        
        images = list()
        c = 1
        for image_path in image_paths:
            if opts['verbose']: print(str(c) + '/' + str(len(image_paths)))
            img = Image.open(image_path)
            if self.opts['out_size'] != -1:
                img = img.resize(self.opts['out_size'], Image.BILINEAR)

            img = np.asarray(img)
            # print(np.max(img))
            img = np.float32(img)/255

            img = np.transpose(img, [2,0,1])
            img = np.expand_dims(img, 0)

            images.append(img)
            c += 1
            
        images = np.concatenate(images, 0)
        images = torch.FloatTensor(images)
        self.images = images
        
    def get_n_train(self):
        return len(self.data['train']['inds'])
    
    def get_n_test(self):
        return len(self.data['test']['inds'])
    
    def get_n_classes(self):
        return self.labels_onehot.shape[1]
        
    def get_images(self, inds, train_or_test):
        dims = list(self.images[0].size())
        dims[0] = len(self.opts['channelInds'])
        
        dims.insert(0, len(inds))
        
        images = torch.zeros(tuple(dims))
        
        c = 0
        for i in inds:
            image = self.images[self.data[train_or_test]['inds'][i]]
            images[c] = image.index_select(0, torch.LongTensor(self.opts['channelInds'])).clone()
            c += 1
        
        # images *= 2
        # images -= 1
        
        return images
    
    def get_rand_images(self, batsize, train_or_test):
        ndat = self.data[train_or_test]['inds']
        
        inds = np.random.choice(ndat, batsize)
        
        return self.get_images(inds, train_or_test)
    
    def get_labels(self, inds, train_or_test, index_or_onehot = 'index'):
        
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