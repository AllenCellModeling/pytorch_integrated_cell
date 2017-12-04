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
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model'
ref_dir = args.parent_dir + os.sep + 'ref_model'

save_dir = args.parent_dir + os.sep + 'analysis' + os.sep + 'inception_scare' + os.sep
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



########
# data #
########

im_class_log_probs_data = {}

# For train or test
for train_or_test in ['test', 'train']:
    ndat = dp.get_n_dat(train_or_test)
    pred_log_probs = np.zeros([ndat,opt.nClasses])
   
    # For each cell in the data split
    for i in tqdm(range(0, 1)):
    #for i in tqdm(range(0, ndat)):

        img_index = dp.data[train_or_test]['inds'][i]
 
        # Load the image
        img_in = dp.get_images([i], train_or_test)
        img_in = Variable(img_in.cuda(gpu_id), volatile=True)

        # pass forward through the model
        z = enc(img_in)
        print(z)
        for w in z:
            print(z.shape)
            print(w.data)
        
#         p = z[0].data.cpu().numpy()
#         pred_log_probs[i,:] = p

#    im_class_log_probs_data[train_or_test] = pred_log_probs

# save test and train preds
# fname = os.path.join(save_dir,'im_class_log_probs_data.pickle')
# with open(fname, 'wb') as handle:
#     pickle.dump(im_class_log_probs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


#############
# generated #
#############

im_class_log_probs_gen = {}

# For train or test
for train_or_test in ['test', 'train']:
    ndat = dp.get_n_dat(train_or_test)
    pred_log_probs = np.zeros([ndat,opt.nClasses])

    # For each cell in the data split
    for i in tqdm(range(0, ndat)):

        # for each image in the real data set
        img_index = dp.data[train_or_test]['inds'][i]
        img_class = dp.get_classes([i], train_or_test, index_or_onehot='onehot')

        img_class_log_onehot = torch.ones(1,opt.nClasses)
        img_class_log_onehot[:,img_class] = 0
        img_class_log_onehot *= -25
        img_class_log_onehot = Variable(img_class_log_onehot)

        z_r = Variable(torch.randn(128).cuda(gpu_id).unsqueeze(0))
        z_s = Variable(torch.randn(128).cuda(gpu_id).unsqueeze(0))

        # generate a fake cell of corresponding class
        z_in = [img_class_log_onehot, z_r, z_s]
        img_in = dec(z_in)

        #pass forward through the model
        z = enc(img_in)
        p = z[0].data.cpu().numpy()
        pred_log_probs[i,:] = p

    im_class_log_probs_gen[train_or_test] = pred_log_probs

# save test and train preds
fname = os.path.join(save_dir,'im_class_log_probs_gen.pickle')
with open(fname, 'wb') as handle:
    pickle.dump(im_class_log_probs_gen, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
