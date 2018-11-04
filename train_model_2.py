import argparse

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

import time
import datetime

import warnings
import json

import integrated_cell as ic
from integrated_cell import model_utils
from integrated_cell.utils import str2bool
from integrated_cell import utils

import shutil
import socket

# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

import pdb

def setup(args):
    if (args['save_parent'] is not None) and (args['save_dir'] is not None):
        raise ValueError('--save_dir and --save_parent are both set. Please choose one or the other.')

    if ((args['train_module'] is not None) and (args['train_module_pt1'] is not None)) or ((args['train_module'] is not None) and (args['train_module_pt2'] is not None)):
        raise ValueError('--train_module and --train_model_pt1 or --train_model_pt2 are both set. Please choose a global train module or specify partial models.')

    the_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if args['save_parent'] is not None:
        args['save_dir'] = os.path.join(args['save_parent'], the_time)

    args['the_time'] = the_time
    args['hostname'] = socket.gethostname()

    if args['data_save_path'] is None:
        args['data_save_path'] = args['save_dir'] + os.sep + 'data.pkl'

    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

    if args['train_module'] is not None:
        args['train_module_pt1'] = args['train_module']
        args['train_module_pt2'] = args['train_module']

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(ID) for ID in args['gpu_ids']])
    args['gpu_ids'] = list(range(0, len(args['gpu_ids'])))

    torch.manual_seed(args['myseed'])
    torch.cuda.manual_seed(args['myseed'])
    np.random.seed(args['myseed'])

    if args['nepochs_pt2'] == -1:
        args['nepochs_pt2'] = args['nepochs']   
    
    return args

def setup_kwargs_data_provider(args):
    kwargs_data_provider = {}
    kwargs_data_provider['name'] = args['kwargs_data_provider']
    kwargs_data_provider['kwargs']['data_path'] = args['data_save_path']
    kwargs_data_provider['kwargs']['batch_size'] = args['batch_size']
    kwargs_data_provider['kwargs']['im_dir'] = args['im_dir']
    kwargs_data_provider['kwargs']['n_dat'] = args['n_dat']
    kwargs_data_provider['kwargs']['channelInds'] = args['channel_inds']
    
    return kwargs_data_provider

def setup_kwargs_network(args):

    kwargs_enc = args['kwargs_enc']
    kwargs_enc['n_channels'] = len(args['channels_pt1'])
    kwargs_enc['n_classes'] = args['n_classes']
    kwargs_enc['n_ref'] = args['n_classes']
    kwargs_enc['n_latent_dim'] = args['nlatentdim']

    kwargs_dec = args['kwargs_dec']
    kwargs_dec['n_channels'] = len(args['channels_pt1'])
    kwargs_dec['n_classes'] = args['n_classes']
    kwargs_dec['n_ref'] = args['n_classes']
    kwargs_dec['n_latent_dim'] = args['nlatentdim']

    kwargs_enc_optim = args['kwargs_enc_optim']
    kwargs_enc_optim['lr'] = args['lrEnc']
    
    kwargs_dec_optim = args['kwargs_enc_optim']
    kwargs_dec_optim['lr'] = args['lrDec']
    
    save_enc_path = '{}/{}.pth'.format(args['save_dir'], 'enc')
    save_dec_path = '{}/{}.pth'.format(args['save_dir'], 'dec')
    
    return args['network_name'], kwargs_enc, kwargs_dec, args['optimizer'], kwargs_enc_optim, kwargs_dec_optim, save_enc_path, save_dec_path

def setup_kwargs_trainer_model(args):
    kwargs_model = {}
    kwargs_model['model_name'] = args['model_name']
    kwargs_model['kwargs'] = pt['kwargs_model']
    kwargs_model['kwargs']['n_epochs'] = args['nepochs']
    kwargs_model['kwargs']['save_dir'] = args['save_dir']
    kwargs_model['kwargs']['save_state_iter'] = args['saveStateIter']
    kwargs_model['kwargs']['save_progress_iter'] = args['saveProgressIter']
    kwargs_model['critRecon'] = args['critRecon']
    kwargs_model['critAdv'] = args['critAdv']
    
    return kwargs_model
    

    

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu id')
parser.add_argument('--myseed', type=int, default=0, help='random seed')
parser.add_argument('--nlatentdim', type=int, default=16, help='number of latent dimensions')
parser.add_argument('--lrEnc', type=float, default=0.0005, help='learning rate for encoder')
parser.add_argument('--lrDec', type=float, default=0.0005, help='learning rate for decoder')

parser.add_argument('--kwargs_enc_optim', type=json.loads, default='{"betas": [0.5, 0.999]}', help='kwargs for encoder optimizer')
parser.add_argument('--kwargs_dec_optim', type=json.loads, default='{"betas": [0.5, 0.999]}', help='kwargs for decoder optimizer')

parser.add_argument('--kwargs_model', type=json.loads, default={}, help='kwargs for the model')

parser.add_argument('--kwargs_enc', type=json.loads, default={}, help='kwargs for the enc')
parser.add_argument('--kwargs_dec', type=json.loads, default={}, help='kwargs for the dec')

parser.add_argument('--kwargs_dp', type=json.loads, default={}, help='kwargs for the data provider')

parser.add_argument('--critRecon', default='BCELoss', help='Loss function for image reconstruction')
parser.add_argument('--critAdv', default='nn.BCEWithLogitsLoss', help='Loss function for advarsaries')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--nepochs', type=int, default=250, help='total number of epochs')
parser.add_argument('--nepochs_pt2', type=int, default=-1, help='total number of epochs')

parser.add_argument('--model_name', default='waaegan', help='name of the model module')
parser.add_argument('--save_dir', type=str, default=None, help='save dir')
parser.add_argument('--save_parent', type=str, default=None, help='parent save directory to save with autogenerated working directory (mutually exclusive to "--save_dir")')
parser.add_argument('--saveProgressIter', type=int, default=1, help='number of iterations between saving progress')
parser.add_argument('--saveStateIter', type=int, default=1, help='number of iterations between saving progress')
parser.add_argument('--data_save_path', default=None, help='save path of data file')
parser.add_argument('--imdir', default='/root/data/release_4_1_17/results_v2/aligned/2D', help='location of images')

parser.add_argument('--ndat', type=int, default=-1, help='Number of data points to use')
parser.add_argument('--optimizer', default='Adam', help='type of optimizer, can be {Adam, RMSprop}')

parser.add_argument('--train_module', default=None, help='training module')
parser.add_argument('--train_module_pt1', default=None, help='training module')
parser.add_argument('--train_module_pt2', default=None, help='training module')

parser.add_argument('--dataProvider', default='DataProvider', help='Dataprovider object')

parser.add_argument('--channels_pt1', nargs='+', type=int, default=[0,2], help='channels to use for part 1')
parser.add_argument('--channels_pt2', nargs='+', type=int, default=[0,1,2], help='channels to use for part 2')

parser.add_argument('--dtype', default='float', help='data type that the dataprovider uses. Only \'float\' supported.')

parser.add_argument('--overwrite_opts', default=False, type=str2bool, help='Overwrite options file')
parser.add_argument('--skip_pt1', default=False, type=str2bool, help='Skip pt 1')

parser.add_argument('--ref_dir', default='ref_model', type=str, help='Directory name for reference model')
parser.add_argument('--struct_dir', default='struct_model', type=str, help='Directory name for structure model')

args = vars(parser.parse_args())

args = setup(args)

save_dir = args['save_dir']

###load the all of the parameters    
args = utils.save_load_dict(save_dir + 'args.json', args, args['overwrite_opts'])
args['save_dir'] = '{}/{}'.format(save_dir, 'ref_model')


###load the dataprovider
args['channel_inds'] = args['channels_pt1']
kwargs_dp = utils.save_load_dict(save_dir + 'args_dp.json', setup_kwargs_data_provider(args), args['overwrite_opts'])
dp = model_utils.load_data_provider(**kwargs_dp)

###load the trainer model
kwargs_trainer_model = utils.save_load_dict(save_dir + 'args_trainer.json', setup_kwargs_trainer_model(args), args['overwrite_opts'])

trainer_module = importlib.import_module("integrated_cell.models." + kwargs_trainer_model['name'])
trainer = trainer_module(**kwargs_trainer_model['kwargs'])

###load the networks
args['n_classes'] = dp.get_n_classes()
args['n_ref'] = dp.get_n_ref()

net_settings = utils.save_load_dict(save_dir + 'args_network.json', setup_kwargs_network(args), args['overwrite_opts'])

network_stuff = trainer_module.load_network(**net_settings)


#######
### TRAIN REFERENCE MODEL
#######
if not os.path.exists(args["save_dir"]):
    os.makedirs(args["save_dir"])

print(trainer)
model = trainer.Model(data_provider = dp, **network_stuff, **kwargs_model)

model.load(args.save_dir)
model.train()
    
#######
### DONE TRAINING REFERENCE MODEL
#######

#######
### TRAIN STRUCTURE MODEL
#######

embeddings_path = args.save_dir + os.sep + 'embeddings.pkl'

if args.skip_pt1:
    embeddings = model_utils.load_embeddings(embeddings_path, None, dp, opt)
else:
    embeddings = model_utils.load_embeddings(embeddings_path, model.enc, dp, opt)

models = None
optimizers = None

dp.embeddings = embeddings
dp.channelInds = args.channels_pt2

args.save_dir = os.path.join(args.save_dir, args.struct_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

kwargs_model = setup_kwargs_model(opt)    

kwargs_model['n_channels'] = len(args.channels_pt2)
kwargs_model['n_classes'] = dp.get_n_classes()
kwargs_model['n_ref'] = args.nlatentdim
kwargs_model['n_epochs'] = args.nepochs_pt2



kwargs_model = utils.load_opts(save_path = '{0}/args.pkl'.format(args.save_dir), 
                         kwargs_model = kwargs_model, 
                         overwrite_opts = args.overwrite_opts)

model_module = importlib.import_module("integrated_cell.models." + args.train_module_pt2)

print(kwargs_model)
model = model_module.Model(data_provider = dp, **kwargs_model)

model.load(args.save_dir)
model.train()

print('Finished Training')

embeddings_path = args.save_dir + os.sep + 'embeddings.pkl'
embeddings = model_utils.load_embeddings(embeddings_path, model.enc, dp, opt)

#######
### DONE TRAINING STRUCTURE MODEL
#######
