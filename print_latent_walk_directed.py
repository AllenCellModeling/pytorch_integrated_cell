#######    
### This function prints off the most likely predicted 
### channels for each of the cells in our dataset
#######

import argparse

import importlib
import numpy as np

import os
import pickle

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils

#have to do this import to be able to use pyplot in the docker image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from IPython import display
import time

import model_utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pdb
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu id')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model'
mito_medoids_file = args.parent_dir + os.sep + 'analysis/z_ref_mito_mediods.csv'

# logger_file = '{0}/logger_tmp.pkl'.format(model_dir)
opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))
print(opt)

opt.gpu_ids = args.gpu_ids

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

data_path = './data_{0}x{1}.pyt'.format(str(opt.imsize), str(opt.imsize))
dp = model_utils.load_data_provider(data_path, opt.imdir, opt.dataProvider)

#######    
### Load REFERENCE MODEL
#######

opt.channelInds = [0, 1, 2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = dp.get_n_classes()
opt.nRef = opt.nlatentdim

models, _, _, _, opt = model_utils.load_model(opt.model_name, opt)

dec = models['dec']
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

from corr_stats import pearsonr, corrcoef

opt.batch_size = 400
gpu_id = opt.gpu_ids[0]

save_parent = opt.save_dir + os.sep + 'latent_walk_directed' + os.sep

if not os.path.exists(save_parent):
    os.makedirs(save_parent)
    

    
nclasses = dp.get_n_classes()
nref = opt.nRef
nlatent = opt.nlatentdim

nsamples = 100

positions = pd.read_csv(mito_medoids_file)
positions = positions[[column for column in positions.columns if 'z_' in column]]
positions = positions.as_matrix()

#Put the post-division mitotic phase at the beginning
positions = np.concatenate([np.expand_dims(positions[7], axis=0), positions[0:7]])


from scipy.interpolate import interp1d

def interp(nsamples, y):
    xs = np.linspace(0, 1, num=nsamples)
    
    positions = np.zeros([nsamples, y.shape[1]])
    
    for i in range(0, y.shape[1]):
        positions[:,i] = interp1d([0,1], y[:,i])(xs)
    
    return positions



positions_final = list()
for i in range(0, len(positions)-1):
    positions_final.append(interp(nsamples, positions[i:i+2])[0:-1])

positions_final = np.concatenate(positions_final)
    
nframes = len(positions_final)    
    


classes = torch.Tensor(nclasses, nclasses).fill_(0).cuda(gpu_id)
for i in range(0,nclasses): classes[i,i] = 1
classes = (classes - 1) * 25
classes = Variable(classes)

struct = Variable(torch.Tensor(nclasses, nlatent).fill_(0).cuda(gpu_id))
    
for i in range(0, nframes):
    path = './{0}/step_{1}.png'.format(save_parent, int(i));
    if os.path.exists(path): os.remove(path)
        
    
def tensor2img(img):
#     colormap = 'hsv'
#     colors = plt.get_cmap(colormap)(np.linspace(0, 1, img.size()[1]+1))
    
    colors = [[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]
    
    # pdb.set_trace()
    
    img = img.numpy()
    im_out = list()
    for i in range(0, img.shape[0]):
        
        img_tmp = img[i]
        
        for j in range(0, len(img_tmp)):
            img_tmp[j] = img_tmp[j]/np.max(img_tmp[j])
        
        img_tmp = np.swapaxes(img_tmp, 1,3)
        im_proj = imgtoprojection(img_tmp, proj_all=True,  colors = colors, global_adjust=True)
        im_proj = np.swapaxes(im_proj, 0, 2)
        
        im_proj = np.flip(im_proj,0)
        im_proj = np.flip(im_proj,1)
        
        im_out.append(im_proj)
    
    
    img1 = np.concatenate(im_out[0:5], 1)
    img2 = np.concatenate(im_out[5:], 1)
    
    img = np.concatenate([img1, img2], 0)

    # if len(img.shape) == 3:
    #     img = np.expand_dims(img, 3)

    # for i in range(0, len(img)):
    #     img[i] = img[i]/np.max(img[i])
    
    # img = np.swapaxes(img, 2,3)
    # img = imgtoprojection(np.swapaxes(img, 1, 3), proj_all=True,  colors = colors, global_adjust=True)

    return img    
    
# init = Variable(torch.Tensor(1, nref).normal_().repeat(nclasses,1).cuda(gpu_id))

stdstep = 0.1

for i in tqdm(range(0, nframes)):
    
    ref = Variable(torch.from_numpy(positions_final[i]).float().repeat(nclasses,1).cuda(gpu_id))
    
    # ref += Variable(torch.Tensor(1, nref).normal_(0,stdstep).repeat(nclasses,1).cuda(gpu_id))
    
    im_out = dec([classes, ref, struct])
    
    im_out = tensor2img(im_out.data.cpu())
    
    scipy.misc.imsave('./{0}/step_{1}.png'.format(save_parent, int(i)), im_out)
    
    
    
    

    
























