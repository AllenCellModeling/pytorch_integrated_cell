import argparse

import DataProvider as DP
import SimpleLogger as SimpleLogger

import importlib
import numpy as np
import scipy.misc
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
from model_utils import set_gpu_recursive

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
parser.add_argument('--clamp_lower', type=float, default=-0.01, help='lower clamp for wasserstein gan')
parser.add_argument('--clamp_upper', type=float, default=0.01, help='upper clamp for wasserstein gan')
parser.add_argument('--model_name', default='waaegan', help='name of the model module')
parser.add_argument('--save_dir', default='./waaegan/', help='save dir')
parser.add_argument('--saveProgressIter', type=int, default=1, help='number of iterations between saving progress')
parser.add_argument('--saveStateIter', type=int, default=10, help='number of iterations between saving progress')
parser.add_argument('--imsize', type=int, default=128, help='pixel size of images used')   
parser.add_argument('--imdir', default='/root/data/release_4_1_17/release_v2/aligned/2D', help='location of images')
parser.add_argument('--latentDistribution', default='gaussian', help='Distribution of latent space, can be {gaussian, uniform}')
parser.add_argument('--ndat', type=int, default=-1, help='Number of data points to use')

parser.add_argument('--optimizer', default='adam', help='type of optimizer, can be {adam, RMSprop}')
parser.add_argument('--train_module', default='waaegan_train', help='training module')
opt = parser.parse_args()
print(opt)

model_provider = importlib.import_module("models." + opt.model_name)
train_module = importlib.import_module("train_modules." + opt.train_module)


if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

DP = importlib.reload(DP)

opts = {}
opts['verbose'] = True
opts['pattern'] = '*.tif_flat.png'
opts['out_size'] = [opt.imsize, opt.imsize]

data_path = './data_' + str(opts['out_size'][0]) + 'x' + str(opts['out_size'][1]) + '.pyt'

if os.path.exists(data_path):
    dp = torch.load(data_path)
else:
    dp = DP.DataProvider(opt.imdir, opts)
    torch.save(dp, data_path)
    
if opt.ndat == -1:
    opt.ndat = dp.get_n_train()    
else:
    dp.set_n_train(opt.ndat)

        
def tensor2img(img):
    
    imresize = list(img.size())
    imresize[1] = 3
    
    img_out = torch.zeros(tuple(imresize))
    
    img_tmp = np.zeros(imresize)
    img_tmp[:, opt.channelInds] = img.numpy()
    img = img_tmp
    
    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    img = np.transpose(img, [0,2,3,1])
    img = np.concatenate(img[:], 1)
    
    return img


def saveProgress(models, dataProvider, logger, zAll, opt):
    enc = models[0]
    dec = models[1]
    
    gpu_id = opt.gpu_ids[0]
    
    enc.train(False)
    dec.train(False)

    x = Variable(dp.get_images(np.arange(0,10),'train')).cuda(gpu_id)
    xHat = dec(enc(x))
    imgX = tensor2img(x.data.cpu())
    imgXHat = tensor2img(xHat.data.cpu())
    imgTrainOut = np.concatenate((imgX, imgXHat), 0)

    x = Variable(dp.get_images(np.arange(0,10),'test')).cuda(gpu_id)
    xHat = dec(enc(x))
    imgX = tensor2img(x.data.cpu())
    imgXHat = tensor2img(xHat.data.cpu())
    imgTestOut = np.concatenate((imgX, imgXHat), 0)

    imgOut = np.concatenate((imgTrainOut, imgTestOut))

    scipy.misc.imsave('./{0}/progress_{1}.png'.format(opt.save_dir, epoch), imgOut)

    enc.train(True)
    dec.train(True)

    # pdb.set_trace()
    # zAll = torch.cat(zAll,0).cpu().numpy()

    pickle.dump(zAll, open('./{0}/embedding_tmp.pkl'.format(opt.save_dir), 'wb'))
    pickle.dump(logger, open('./{0}/logger_tmp.pkl'.format(opt.save_dir), 'wb'))
    
def saveState(models, optimizers, logger, zAll, opt):
#         for saving and loading see:
#         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718
    enc = models[0]    
    dec = models[1]
    encD = models[2]
    decD = models[3]
    
    optEnc = optimizers[0]
    optDec = optimizers[1]
    optEncD = optimizers[2]
    optDecD = optimizers[3]
    
    gpu_id = opt.gpu_ids[0]
    
    enc = enc.cpu()
    dec = dec.cpu()
    encD = encD.cpu()
    decD = decD.cpu()

    optEnc.state = set_gpu_recursive(optEnc.state, -1)
    optDec.state = set_gpu_recursive(optDec.state, -1)
    optEncD.state = set_gpu_recursive(optEncD.state, -1)
    optDecD.state = set_gpu_recursive(optDecD.state, -1)

    torch.save(enc.state_dict(), './{0}/enc.pth'.format(opt.save_dir))
    torch.save(dec.state_dict(), './{0}/dec.pth'.format(opt.save_dir))
    torch.save(encD.state_dict(), './{0}/encD.pth'.format(opt.save_dir))
    torch.save(decD.state_dict(), './{0}/decD.pth'.format(opt.save_dir))

    torch.save(optEnc.state_dict(), './{0}/optEnc.pth'.format(opt.save_dir))
    torch.save(optDec.state_dict(), './{0}/optDec.pth'.format(opt.save_dir))
    torch.save(optEncD.state_dict(), './{0}/optEncD.pth'.format(opt.save_dir))
    torch.save(optDecD.state_dict(), './{0}/optDecD.pth'.format(opt.save_dir))

    enc.cuda(gpu_id)
    dec.cuda(gpu_id)
    encD.cuda(gpu_id)
    decD.cuda(gpu_id)

    optEnc.state = set_gpu_recursive(optEnc.state, gpu_id)
    optDec.state = set_gpu_recursive(optDec.state, gpu_id)
    optEncD.state = set_gpu_recursive(optEncD.state, gpu_id)
    optDecD.state = set_gpu_recursive(optDecD.state, gpu_id)


    pickle.dump(zAll, open('./{0}/embedding.pkl'.format(opt.save_dir), 'wb'))
    pickle.dump(logger, open('./{0}/logger.pkl'.format(opt.save_dir), 'wb'))
    
        
opt.channelInds = [0,2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
models, optimizers, criterions, logger, opt = train_module.load(model_provider, opt)

start_iter = len(logger.log['iter'])


zAll = list()

for this_iter in range(start_iter, math.ceil(opt.ndat/opt.batch_size)*opt.nepochs):
    epoch = np.floor(this_iter/(opt.ndat/opt.batch_size))
    epoch_next = np.floor((this_iter+1)/(opt.ndat/opt.batch_size))
    
    start = time.time()
    
    errors, zfake = train_module.iteration(models, optimizers, criterions, dp, this_iter, opt)
    
    zAll.append(zfake)
    
    stop = time.time()
    deltaT = stop-start
    
    
    logger.add((epoch, this_iter) + errors +(deltaT,))
    
    if epoch != epoch_next and ((epoch % opt.saveProgressIter) == 0 or (this_iter % opt.saveStateIter) == 0):
        zAll = torch.cat(zAll,0).cpu().numpy()
        
        if (epoch % opt.saveProgressIter) == 0:
            saveProgress(models, dp, logger, zAll, opt)
        
        if (epoch % opt.saveStateIter) == 0:
            saveState(models, optimizers, logger, zAll, opt)
            
        zAll = list()
          

print('Finished Training')



