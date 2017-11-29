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

import model_utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

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

parser.add_argument('--critRecon', default='BCELoss', help='Loss function for image reconstruction')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--nepochs', type=int, default=250, help='total number of epochs')
parser.add_argument('--nepochs_pt2', type=int, default=-1, help='total number of epochs')

parser.add_argument('--model_name', default='waaegan', help='name of the model module')
parser.add_argument('--save_dir', default='./test_waaegan/waaegan/', help='save dir')
parser.add_argument('--saveProgressIter', type=int, default=1, help='number of iterations between saving progress')
parser.add_argument('--saveStateIter', type=int, default=10, help='number of iterations between saving progress')
parser.add_argument('--data_save_path', default=None, help='save path of data file')   
parser.add_argument('--imdir', default='/root/data/release_4_1_17/results_v2/aligned/2D', help='location of images')
parser.add_argument('--latentDistribution', default='gaussian', help='Distribution of latent space, can be {gaussian, uniform}')
parser.add_argument('--ndat', type=int, default=-1, help='Number of data points to use')
parser.add_argument('--optimizer', default='adam', help='type of optimizer, can be {adam, RMSprop}')
parser.add_argument('--train_module', default='waaegan_train', help='training module')
parser.add_argument('--noise', type=float, default=0, help='Noise added to the decD')
parser.add_argument('--dataProvider', default='DataProvider', help='Dataprovider object')

parser.add_argument('--channels_pt1', nargs='+', type=int, default=[0,2], help='channels to use for part 1')
parser.add_argument('--channels_pt2', nargs='+', type=int, default=[0,1,2], help='channels to use for part 2')

parser.add_argument('--dtype', default='float', help='data type that the dataprovider uses. Only \'float\' supported.')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(ID) for ID in opt.gpu_ids])
opt.gpu_ids = list(range(0, len(opt.gpu_ids)))

opt.save_parent = opt.save_dir

if opt.data_save_path is None:
    opt.data_save_path = opt.save_dir + os.sep + 'data.pyt'

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)
    
if opt.nepochs_pt2 == -1:
    opt.nepochs_pt2 = opt.nepochs

dp = model_utils.load_data_provider(opt.data_save_path, opt.imdir, opt.dataProvider)
    
if opt.ndat == -1:
    opt.ndat = dp.get_n_dat('train')

iters_per_epoch = np.ceil(opt.ndat/opt.batch_size)    
            
#######    
### TRAIN REFERENCE MODEL
#######

opt.save_dir = opt.save_parent + os.sep + 'ref_model'
if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

opt.channelInds = opt.channels_pt1
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = 0
opt.nRef = 0

try:    
    train_module = importlib.import_module("train_modules." + opt.train_module)
    train_module = train_module.trainer(dp, opt)
except:
    pass    

pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))

models, optimizers, criterions, logger, opt = model_utils.load_model(opt.model_name, opt)

start_iter = len(logger.log['iter'])
zAll = list()
for this_iter in range(start_iter, math.ceil(iters_per_epoch)*opt.nepochs):
    opt.iter = this_iter
    
    epoch = np.floor(this_iter/iters_per_epoch)
    epoch_next = np.floor((this_iter+1)/iters_per_epoch)
    
    start = time.time()

    errors, zfake = train_module.iteration(**models, **optimizers, **criterions, dataProvider=dp, opt=opt)
    
    zAll.append(zfake)
    
    stop = time.time()
    deltaT = stop-start
    
    logger.add((epoch, this_iter) + errors +(deltaT,))
    
    if model_utils.maybe_save(epoch, epoch_next, models, optimizers, logger, zAll, dp, opt):
        zAll = list()

#######
### DONE TRAINING REFERENCE MODEL
#######

#######    
### TRAIN STRUCTURE MODEL
#######

embeddings_path = opt.save_dir + os.sep + 'embeddings.pkl'
embeddings = model_utils.load_embeddings(embeddings_path, models['enc'], dp, opt) 

models = None
optimizers = None
    
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
    
opt.channelInds = opt.channels_pt2
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = dp.get_n_classes()
opt.nRef = opt.nlatentdim

try:
    train_module = None
    train_module = importlib.import_module("train_modules." + opt.train_module)
    train_module = train_module.trainer(dp, opt)
except:
    pass    

pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))

models, optimizers, criterions, logger, opt = model_utils.load_model(opt.model_name, opt)

start_iter = len(logger.log['iter'])

zAll = list() 
for this_iter in range(start_iter, math.ceil(iters_per_epoch)*opt.nepochs_pt2):
    opt.iter = this_iter
    
    epoch = np.floor(this_iter/(iters_per_epoch))
    epoch_next = np.floor((this_iter+1)/(iters_per_epoch))
    
    start = time.time()
    
    errors, zfake = train_module.iteration(**models, **optimizers, **criterions, dataProvider=dp, opt=opt)
    
    zAll.append(zfake)
    
    stop = time.time()
    deltaT = stop-start
    
    logger.add((epoch, this_iter) + errors +(deltaT,))
    
    if model_utils.maybe_save(epoch, epoch_next, models, optimizers, logger, zAll, dp, opt):
        zAll = list()
            
print('Finished Training')

embeddings_path = opt.save_dir + os.sep + 'embeddings.pkl'
embeddings = model_utils.load_embeddings(embeddings_path, models['enc'], dp, opt) 

#######    
### DONE TRAINING STRUCTURE MODEL
#######

