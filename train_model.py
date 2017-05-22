import argparse

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

#have to do this import to be able to use pyplot in the docker image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from IPython import display
import time
from model_utils import set_gpu_recursive, load_model, save_state, save_progress, get_latent_embeddings, maybe_save

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--Diters', type=int, default=5, help='niters for the encD')
parser.add_argument('--DitersAlt', type=int, default=100, help='niters for the encD')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu id')
parser.add_argument('--myseed', type=int, default=0, help='random seed')
parser.add_argument('--nlatentdim', type=int, default=16, help='number of latent dimensions')
parser.add_argument('--lrEnc', type=float, default=0.0005, help='learning rate for encoder')
parser.add_argument('--lrDec', type=float, default=0.0005, help='learning rate for decoder')
parser.add_argument('--lrEncD', type=float, default=0.00005, help='learning rate for encD')
parser.add_argument('--lrDecD', type=float, default=0.00005, help='learning rate for decD')
parser.add_argument('--encDRatio', type=float, default=5E-3, help='scalar applied to the update gradient from encD')
parser.add_argument('--decDRatio', type=float, default=1E-4, help='scalar applied to the update gradient from decD')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--nepochs', type=int, default=250, help='total number of epochs')
parser.add_argument('--nepochs_pt2', type=int, default=-1, help='total number of epochs')
parser.add_argument('--clamp_lower', type=float, default=-0.01, help='lower clamp for wasserstein gan')
parser.add_argument('--clamp_upper', type=float, default=0.01, help='upper clamp for wasserstein gan')
parser.add_argument('--model_name', default='waaegan', help='name of the model module')
parser.add_argument('--save_dir', default='./test_waaegan/waaegan/', help='save dir')
parser.add_argument('--saveProgressIter', type=int, default=1, help='number of iterations between saving progress')
parser.add_argument('--saveStateIter', type=int, default=10, help='number of iterations between saving progress')
parser.add_argument('--imsize', type=int, default=128, help='pixel size of images used')   
parser.add_argument('--imdir', default='/root/data/release_4_1_17/release_v2/aligned/2D', help='location of images')
parser.add_argument('--latentDistribution', default='gaussian', help='Distribution of latent space, can be {gaussian, uniform}')
parser.add_argument('--ndat', type=int, default=-1, help='Number of data points to use')
parser.add_argument('--optimizer', default='adam', help='type of optimizer, can be {adam, RMSprop}')
parser.add_argument('--train_module', default='waaegan_train', help='training module')
parser.add_argument('--noise', type=float, default=0, help='Noise added to the decD')
parser.add_argument('--dataProvider', default='DataProvider', help='Dataprovider object')

opt = parser.parse_args()
print(opt)

opt.save_parent = opt.save_dir

DP = importlib.import_module("data_providers." + opt.dataProvider)
model_provider = importlib.import_module("models." + opt.model_name)
train_module = importlib.import_module("train_modules." + opt.train_module)

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)
    
if opt.nepochs_pt2 == -1:
    opt.nepochs_pt2 = opt.nepochs

pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))

opts = {}
opts['verbose'] = True
opts['pattern'] = '*.tif_flat.png'
opts['out_size'] = [opt.imsize, opt.imsize]

data_path = './data_{0}x{1}.pyt'.format(str(opts['out_size'][0]), str(opts['out_size'][1]))
if os.path.exists(data_path):
    dp = torch.load(data_path)
else:
    dp = DP.DataProvider(opt.imdir, opts)
    torch.save(dp, data_path)
    
if opt.ndat == -1:
    opt.ndat = dp.get_n_dat('train')    

iters_per_epoch = np.ceil(opt.ndat/opt.batch_size)    
            
#######    
### TRAIN REFERENCE MODEL
#######

opt.save_dir = opt.save_parent + os.sep + 'ref_model'
if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

opt.channelInds = [0,2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = 0
opt.nRef = 0

try:
    train_module = train_module.trainer(dp, opt)
except:
    pass    

pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))

models, optimizers, criterions, logger, opt = load_model(model_provider, opt)

start_iter = len(logger.log['iter'])
zAll = list()
for this_iter in range(start_iter, math.ceil(iters_per_epoch)*opt.nepochs):
    epoch = np.floor(this_iter/iters_per_epoch)
    epoch_next = np.floor((this_iter+1)/iters_per_epoch)
    
    start = time.time()
    
    errors, zfake = train_module.iteration(**models, **optimizers, **criterions, dataProvider=dp, opt=opt)
    
    zAll.append(zfake)
    
    stop = time.time()
    deltaT = stop-start
    
    logger.add((epoch, this_iter) + errors +(deltaT,))
    
    if maybe_save(epoch, epoch_next, models, optimizers, logger, zAll, dp, opt):
        zAll = list()

#######
### DONE TRAINING REFERENCE MODEL
#######

#######    
### TRAIN STRUCTURE MODEL
#######

embeddings_path = opt.save_dir + os.sep + 'embeddings.pkl'
if os.path.exists(embeddings_path):
    embeddings = torch.load(embeddings_path)
else:
    embeddings = get_latent_embeddings(models['enc'], dp, opt)
    torch.save(embeddings, embeddings_path)

def get_ref(self, inds, train_or_test='train'):
    inds = torch.LongTensor(inds)
    return self.embeddings[train_or_test][inds]

dp.embeddings = embeddings

# do this thing to bind the get_ref method to the dataprovider object
import types  
dp.get_ref = types.MethodType(get_ref, dp)
            
opt.save_dir = opt.save_parent + os.sep + 'struct_model'
if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)
    
opt.channelInds = [0, 1, 2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = dp.get_n_classes()
opt.nRef = opt.nlatentdim

pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))

models, optimizers, criterions, logger, opt = load_model(model_provider, opt)

start_iter = len(logger.log['iter'])

zAll = list() 
for this_iter in range(start_iter, math.ceil(iters_per_epoch)*opt.nepochs_pt2):
    epoch = np.floor(this_iter/(iters_per_epoch))
    epoch_next = np.floor((this_iter+1)/(iters_per_epoch))
    
    start = time.time()
    
    errors, zfake = train_module.iteration(**models, **optimizers, **criterions, dataProvider=dp, opt=opt)
    
    zAll.append(zfake)
    
    stop = time.time()
    deltaT = stop-start
    
    logger.add((epoch, this_iter) + errors +(deltaT,))
    
    if maybe_save(epoch, epoch_next, models, optimizers, logger, zAll, dp, opt):
        zAll = list()
            
print('Finished Training')

#######    
### DONE TRAINING STRUCTURE MODEL
#######

