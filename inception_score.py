#######    
### This function prints off the inception score
### for both the input images and generated images
#######

import argparse

import importlib
import numpy as np
import pandas as pd

import os
import pickle

import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#have to do this import to be able to use pyplot in the docker image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from IPython import display

import model_utils


import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pdb
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='gpu id')
parser.add_argument('--batch_size', type=int, default=200, help='gpu id')
parser.add_argument('--overwrite', type=bool, default=False, help='overwrite?')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model'
ref_dir = args.parent_dir + os.sep + 'ref_model'

save_dir = args.parent_dir + os.sep + 'analysis' + os.sep + 'inception_score' + os.sep
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
        

# logger_file = '{0}/logger_tmp.pkl'.format(model_dir)
opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))
print(opt)

opt.gpu_ids = args.gpu_ids
gpu_id = opt.gpu_ids[0]
torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

dp = model_utils.load_data_provider(opt.data_save_path, opt.imdir, opt.dataProvider)

#######    
### Load REFERENCE MODEL
#######

opt.channelInds = [0, 1, 2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = dp.get_n_classes()
opt.nRef = opt.nlatentdim

models, _, _, _, opt = model_utils.load_model(opt.model_name, opt)

opt.batch_size = args.batch_size

enc = models['enc']
dec = models['dec']
enc.train(False)
dec.train(False)

models = None
optimizers = None

print('Done loading model.')

#######    
### Main Loop
#######

import pdb
from aicsimage.io import omeTifWriter
from imgToProjection import imgtoprojection
import PIL.Image
from aicsimage.io import omeTifWriter

import scipy.misc
import pandas as pd



##########
## data ##
##########
fname = os.path.join(save_dir,'im_class_log_probs_data.pickle')

if os.path.exists(fname) and not args.overwrite:
    pass
else:

    im_class_log_probs = {}

    # For train or test
    for train_or_test in ['test', 'train']:
        ndat = dp.get_n_dat(train_or_test)
        inds = np.arange(0, ndat)    

        pred_log_probs = np.zeros([ndat,opt.nClasses])

        iter_struct = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)] 

        # For each cell in the data split
    #     for i in tqdm(range(0, 1)):
        for i in tqdm(iter_struct, desc='data, ' + train_or_test):

            # Load the image
            img_in = dp.get_images(i, train_or_test)
            img_in = Variable(img_in.cuda(gpu_id), volatile=True)

            # pass forward through the model
            z = enc(img_in)
            
            p = z[0].data.cpu().numpy()
            pred_log_probs[i,:] = p

        im_class_log_probs[train_or_test] = pred_log_probs

    # save test and train preds

    with open(fname, 'wb') as handle:
        pickle.dump(im_class_log_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)


#############
# autoencoded #
#############

fname = os.path.join(save_dir,'im_class_log_probs_autoencode.pickle')

if os.path.exists(fname) and not args.overwrite:
    pass
else:

    im_class_log_probs = {}

    # For train or test
    for train_or_test in ['test', 'train']:
        ndat = dp.get_n_dat(train_or_test)
        inds = np.arange(0, ndat)      

        pred_log_probs = np.zeros([ndat,opt.nClasses])

        iter_struct = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)] 

        for i in tqdm(iter_struct, desc='autoencode, ' + train_or_test):

            # Load the image
            img_in = dp.get_images(i, train_or_test)
            img_in = Variable(img_in.cuda(gpu_id), volatile=True)

            # pass forward through the model
            z = enc(dec(enc(img_in)))

            p = z[0].data.cpu().numpy()
            pred_log_probs[i,:] = p

        im_class_log_probs[train_or_test] = pred_log_probs

    # save test and train preds

    with open(fname, 'wb') as handle:
        pickle.dump(im_class_log_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

#############
# autoencoded #
#############
fname = os.path.join(save_dir,'im_class_log_probs_gen.pickle')

if os.path.exists(fname) and not args.overwrite:
    pass
else:

    im_class_log_probs = {}

    # For train or test
    for class_id in range(0, dp.get_n_classes()):
    #     ndat = dp.get_n_dat(train_or_test)
        ndat = 2000
        inds = np.arange(0, ndat)    

        pred_log_probs = np.zeros([ndat,opt.nClasses])

        iter_struct = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)] 

        # For each cell in the data split
        for i in tqdm(iter_struct, desc='gen, ' + str(class_id+1) + os.sep + str(dp.get_n_classes()) ):

            # create the log onehot class vector
            classes = Variable(torch.Tensor(ndat, opt.nClasses).fill_(-25).cuda(gpu_id), volatile=True)
            classes[:, class_id] = 0

            # sample random latent space vectors
            ref = Variable(torch.Tensor(ndat, opt.nRef).normal_().cuda(gpu_id), volatile=True)
            struct = Variable(torch.Tensor(ndat, opt.nRef).normal_().cuda(gpu_id), volatile=True)

            # generate a fake cell of corresponding class
            img_in = dec([classes, ref, struct])

            # pass forward through the model
            z = enc(img_in)
            p = z[0].data.cpu().numpy()
            pred_log_probs[i,:] = p

        im_class_log_probs[train_or_test] = pred_log_probs

    # save test and train preds

    with open(fname, 'wb') as handle:
        pickle.dump(im_class_log_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
