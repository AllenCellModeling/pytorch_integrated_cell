#######    
### This function prints off the most likely predicted 
### channels for each of the cells in our dataset
#######

#######    
### Load the Model Parts
#######

import SimpleLogger as SimpleLogger

import importlib
import numpy as np

import os
import pickle

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils

import model_utils

from tqdm import tqdm

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='gpu id')
parser.add_argument('--batch_size', type=int, default=400, help='batch_size')
parser.add_argument('--overwrite', type=bool, default=False, help='overwrite existing results')
parser.add_argument('--model_dir', default='struct_model', help='Model component direcoty')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + args.model_dir 

opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))
print(opt)

opt.gpu_ids = args.gpu_ids

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

models, optimizers, _, _, opt = model_utils.load_model(opt.model_name, opt)

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
from IPython.core.display import display
import PIL.Image
import matplotlib.pyplot as plt
import scipy.misc

import pandas as pd

gpu_id = opt.gpu_ids[0]

colormap = 'hsv'
colors = plt.get_cmap(colormap)(np.linspace(0, 1, 4))

# [magenta, yellow, cyan]
# colors = [[1, 0, 1], [1, 1, 0], [0, 1, 1]]

px_size = [1,1,1]

train_or_test_split = ['test', 'train']

img_paths_all = list()

save_parent = args.parent_dir + os.sep + 'analysis' + os.sep + 'proj_single_channel'
save_out_table = save_parent + os.sep + 'list_of_images.csv'

column_names = ['orig', 'recon'] + ['pred_' + name for name in dp.label_names] + ['train_or_test', 'orig_struct', 'img_index']

if not os.path.exists(save_parent):
    os.makedirs(save_parent)

def convert_image(img):
    img = img.data[0].cpu().numpy()
    img = np.transpose(img, (3, 0, 1, 2))

    return img
    
def img2projection(img):
    img = convert_image(img)
    img = np.transpose(img, (1,0,2,3))
    img = imgtoprojection(img, proj_all=True, colors = colors, global_adjust=True)
    img = np.transpose(img, (1,2,0))
    
    return img

# For train or test
for train_or_test in train_or_test_split:
    ndat = dp.get_n_dat(train_or_test)
    # For each cell in the data split
    for i in tqdm(range(0, ndat)):
        
        img_index = dp.data[train_or_test]['inds'][i]
        img_class = dp.image_classes[img_index]
        img_name = os.path.basename(dp.get_image_paths([i], train_or_test)[0])[0:-3]
        
        save_dir = save_parent + os.sep + train_or_test + os.sep + img_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        pred_all_path = save_dir + os.sep + 'img' + str(img_index) + '_' + img_class + '-pred_all.png'
        
        if os.path.exists(pred_all_path) and not args.overwrite:
            continue
        
        #Load the image
        img_in = dp.get_images([i], train_or_test)
        img_in = Variable(img_in.cuda(gpu_id), volatile=True)
        
        #pass forward through the model
        z = enc(img_in)
        img_recon = dec(z)
        
        pred_imgs = list()
        img_paths = list()
        
        ### Original
        img = img2projection(img_in)
        img_path_base = save_dir + os.sep + 'img' + str(img_index) + '_orig'

        img_name = img_path_base + '_memb.png'
        scipy.misc.imsave(img_name, img[:,:,0])

        img_name = img_path_base + '_' + img_class + '.png'
        scipy.misc.imsave(img_name, img[:,:,1])

        img_name = img_path_base + '_nuc.png'
        scipy.misc.imsave(img_name, img[:,:,2])
        
        ### Reconstruction
        img = img2projection(img_recon)
        img_path_base = save_dir + os.sep + 'img' + str(img_index) + '_recon'

        img_name = img_path_base + '_memb.png'
        scipy.misc.imsave(img_name, img[:,:,0])

        img_name = img_path_base + '_' + img_class + '.png'
        scipy.misc.imsave(img_name, img[:,:,1])

        img_name = img_path_base + '_nuc.png'
        scipy.misc.imsave(img_name, img[:,:,2])

      
        var_classes = (np.identity(dp.get_n_classes()) - 1) * 25
        var_struct = np.zeros([dp.get_n_classes(), z[-1].size()[1]])
        
        z[0] = Variable(torch.Tensor(var_classes).cuda(gpu_id), volatile=True)
        z[1] = z[1].repeat(dp.get_n_classes(), 1)
        z[2] = Variable(torch.Tensor(var_struct).cuda(gpu_id), volatile=True)
        
        img_pred = dec(z)
        
        img_path_base = save_dir + os.sep + 'img' + str(img_index) + '_pred_'
        
        for j, class_name in zip(range(0, dp.get_n_classes()), dp.label_names):
            img_name = img_path_base + class_name + '.png'
            
            img = img2projection(img_pred[[j]])
            scipy.misc.imsave(img_name, img[:,:,1])
        
  
