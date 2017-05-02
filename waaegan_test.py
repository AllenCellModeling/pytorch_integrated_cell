import argparse

import DataProvider as DP
import SimpleLogger as SimpleLogger

import importlib
import numpy as np
import scipy.misc
import os
import pickle

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

import models.waaegan as waaegan

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--Diters', type=int, default=5, help='niters for the encD')
parser.add_argument('--DitersAlt', type=int, default=100, help='niters for the encD')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu id')
parser.add_argument('--myseed', type=int, default=0, help='random seed')
parser.add_argument('--nlatentdim', type=int, default=16, help='number of latent dimensions')
parser.add_argument('--lrEnc', type=float, default=0.00005, help='learning rate for encoder')
parser.add_argument('--lrDec', type=float, default=0.00005, help='learning rate for decoder')
parser.add_argument('--lrEncD', type=float, default=0.00005, help='learning rate for encD')
parser.add_argument('--lrDecD', type=float, default=0.00005, help='learning rate for decD')
parser.add_argument('--encDRatio', type=float, default=1E-3, help='scalar applied to the update gradient from encD')
parser.add_argument('--decDRatio', type=float, default=1E-3, help='scalar applied to the update gradient from decD')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--nepochs', type=int, default=250, help='total number of epochs')
parser.add_argument('--clamp_lower', type=float, default=-0.01, help='lower clamp for wasserstein gan')
parser.add_argument('--clamp_upper', type=float, default=0.01, help='upper clamp for wasserstein gan')
parser.add_argument('--save_dir', default='./waaegan/', help='save dir')
parser.add_argument('--saveProgressIter', type=int, default=1, help='number of iterations between saving progress')
parser.add_argument('--saveStateIter', type=int, default=10, help='number of iterations between saving progress')
parser.add_argument('--imsize', type=int, default=128, help='pixel size of images used')   
parser.add_argument('--imdir', default='/root/images/release_4_1_17_2D', help='location of images')
parser.add_argument('--latentDistribution', default='gaussian', help='Distribution of latent space, can be {gaussian, uniform}')

opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

DP = importlib.reload(DP)

np.random.seed(opt.myseed)

opts = {}
opts['verbose'] = True
opts['pattern'] = '*.tif_flat.png'
opts['out_size'] = [opt.imsize, opt.imsize]

data_path = './data_' + str(opts['out_size'][0]) + 'x' + str(opts['out_size'][1]) + '.pyt'


if opt.latentDistribution == 'uniform':
    def latentSample (batsize, nlatentdim): return torch.Tensor(batsize, nlatentdim).uniform_(-1, 1)
elif opt.latentDistribution == 'gaussian':
    def latentSample (batsize, nlatentdim): return torch.Tensor(batsize, nlatentdim).normal_()

if os.path.exists(data_path):
    dp = torch.load(data_path)
else:
    dp = DP.DataProvider(opt.imdir, opts)
    torch.save(dp, data_path)

def tensor2img(img):
    img = img.numpy()
    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    img = np.transpose(img, [0,2,3,1])
    img = np.concatenate(img[:], 1)
    return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
    
enc = waaegan.Enc(opt.nlatentdim, opt.imsize, opt.gpu_ids)
dec = waaegan.Dec(opt.nlatentdim, opt.imsize, opt.gpu_ids)
encD = waaegan.EncD(opt.nlatentdim, opt.gpu_ids)
decD = waaegan.DecD(1, opt.imsize, opt.gpu_ids)

enc.apply(weights_init)
dec.apply(weights_init)
encD.apply(weights_init)
decD.apply(weights_init)

gpu_id = opt.gpu_ids[0]
nlatentdim = opt.nlatentdim

enc.cuda(gpu_id)
dec.cuda(gpu_id)
encD.cuda(gpu_id)
decD.cuda(gpu_id)

optEnc = optim.RMSprop(enc.parameters(), lr=opt.lrEnc)
optDec = optim.RMSprop(dec.parameters(), lr=opt.lrDec)
optEncD = optim.RMSprop(encD.parameters(), lr=opt.lrEncD)
optDecD = optim.RMSprop(decD.parameters(), lr=opt.lrDecD)

logger = SimpleLogger.SimpleLogger(('epoch', 'iter', 'reconLoss', 'minimaxEncDLoss', 'encDLoss', 'minimaxDecDLoss', 'decDLoss', 'time'), '[%d][%d] reconLoss: %.6f mmEncD: %.6f encD: %.6f mmDecD: %.6f decD: %.6f time: %.2f')

this_epoch = 1
iteration = 0
if os.path.exists('./{0}/enc.pth'.format(opt.save_dir)):
    enc.load_state_dict(torch.load('./{0}/enc.pth'.format(opt.save_dir)))
    dec.load_state_dict(torch.load('./{0}/dec.pth'.format(opt.save_dir)))
    encD.load_state_dict(torch.load('./{0}/encD.pth'.format(opt.save_dir)))
    decD.load_state_dict(torch.load('./{0}/decD.pth'.format(opt.save_dir)))

    optEnc.load_state_dict(torch.load('./{0}/optEnc.pth'.format(opt.save_dir)))
    optDec.load_state_dict(torch.load('./{0}/optDec.pth'.format(opt.save_dir)))
    optEncD.load_state_dict(torch.load('./{0}/optEncD.pth'.format(opt.save_dir)))
    optDecD.load_state_dict(torch.load('./{0}/optDecD.pth'.format(opt.save_dir)))

    opt = pickle.load(open( '{0}/opt.pkl'.format(opt.save_dir), "rb" ))
    logger = pickle.load(open( '{0}/logger.pkl'.format(opt.save_dir), "rb" ))

    this_epoch = max(logger.log['epoch']) + 1
    iteration = max(logger.log['iter'])

criterion = nn.BCELoss()

# optEnc = optim.Adam(enc.parameters(), lr=opt.lrEnc, betas=(0.5, 0.9))
# optDec = optim.Adam(dec.parameters(), lr=opt.lrDec, betas=(0.5, 0.9))
# optEncD = optim.Adam(encD.parameters(), lr=opt.lrEncD, betas=(0.5, 0.9))
# optDecD = optim.Adam(decD.parameters(), lr=opt.lrDecD, betas=(0.5, 0.9))

ndat = dp.get_n_train()
ndat = 1000


one = torch.FloatTensor([1]).cuda(gpu_id)
mone = one * -1

for epoch in range(this_epoch, opt.nepochs+1): # loop over the dataset multiple times

    
    rand_inds = np.random.permutation(ndat)
    inds = (rand_inds[i:i+opt.batch_size] for i in range(0, len(rand_inds), opt.batch_size))
    
    zAll = list()
    
    c = 0
    for i in inds:
        start = time.time()
        
        c += 1
        iteration += 1
        
        batsize = len(i)

        yReal = Variable(torch.ones(batsize)).cuda(gpu_id)
        yFake = Variable(torch.zeros(batsize)).cuda(gpu_id)
        
        ###update the discriminator
        #maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))
        for p in encD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for p in decD.parameters():
            p.requires_grad = True

        # train the discriminator Diters times
        
        # if epoch == 6:
        #     pdb.set_trace()
            
        
        if epoch <= 5 or (iteration % 25) == 0:
            Diters = opt.DitersAlt
        else:
            Diters = opt.Diters
        

        rand_inds_encD = np.random.permutation(ndat)
        niter = len(range(0, len(rand_inds_encD), opt.batch_size))
        inds_encD = (rand_inds_encD[i:i+opt.batch_size] for i in range(0, len(rand_inds_encD), opt.batch_size))
        
        j = 0
        while j < Diters and j < niter:
            j += 1

            # clamp parameters to a cube
            for p in encD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
            
            for p in decD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
                
            x = Variable(dp.get_images(next(inds_encD),'train')).cuda(gpu_id)
            
            zFake = enc(x)
            #pick a distribution that is obvious when you plot it
            # zReal = Variable(torch.Tensor(batsize, nlatentdim).uniform_(-2, 2)).cuda(gpu_id)
            zReal = Variable(latentSample(batsize, nlatentdim)).cuda(gpu_id)
            
            optEnc.zero_grad()
            optDec.zero_grad()
            optEncD.zero_grad()
            optDecD.zero_grad()

            # train with real
            errEncD_real = encD(zReal)
            errEncD_real.backward(one, retain_variables=True)

            # train with fake
            errEncD_fake = encD(zFake)
            errEncD_fake.backward(mone, retain_variables=True)
            encDLoss = errEncD_real - errEncD_fake
            optEncD.step()
            
            
            xHat = dec(zFake.detach())
            
            errDecD_real = decD(x)
            errDecD_real.backward(one, retain_variables=True)
            
            errDecD_fake = decD(xHat)
            errDecD_fake.backward(mone, retain_variables=True)
            
            decDLoss = errDecD_real - errDecD_fake
            optDecD.step()

            
            
        optEnc.zero_grad()
        optDec.zero_grad()
        optEncD.zero_grad()  
        optDecD.zero_grad()          
        
#         x = Variable(dp.get_images(i, 'train')).cuda(gpu_id)
        
        zFake = enc(x)
        xHat = dec(zFake)
        optEnc.step()
        optEnc.step()
        
        optEnc.zero_grad()
        optDec.zero_grad()
    
        reconLoss = criterion(xHat, x)
        reconLoss.backward(retain_variables=True)
        
        for p in encD.parameters():
            p.requires_grad = False
            
        for p in decD.parameters():
            p.requires_grad = False
        
        minimaxEncDLoss = encD(zFake)
        minimaxEncDLoss.backward(one*opt.encDRatio, retain_variables=True)

        optEnc.step()
        
        minimaxDecDLoss = decD(xHat)
        minimaxDecDLoss.backward(one*opt.decDRatio, retain_variables=True)
        
        zReal = Variable(latentSample(batsize, nlatentdim)).cuda(gpu_id)
        xHat = dec(zReal.detach())
        
        minimaxDecDLoss2 = decD(xHat)
        minimaxDecDLoss2.backward(one*opt.decDRatio, retain_variables=True)
        
        optDec.step()

        zAll.append(zFake.data)
        
        stop = time.time()
        deltaT = stop-start
        
        logger.add((epoch, iteration, reconLoss.data[0], minimaxEncDLoss.data[0], encDLoss.data[0], minimaxDecDLoss.data[0], decDLoss.data[0], deltaT))


    
    if (epoch % opt.saveProgressIter) == 0:

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
        
        zAll = torch.cat(zAll,0).cpu().numpy()
        
        pickle.dump(zAll, open('./{0}/embedding.pkl'.format(opt.save_dir), 'wb'))
        pickle.dump(logger, open('./{0}/logger.pkl'.format(opt.save_dir), 'wb'))

    if (epoch % opt.saveStateIter) == 0:
#         for saving and loading see:
#         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718
        torch.save(enc.state_dict(), './{0}/enc.pth'.format(opt.save_dir))
        torch.save(dec.state_dict(), './{0}/dec.pth'.format(opt.save_dir))
        torch.save(encD.state_dict(), './{0}/encD.pth'.format(opt.save_dir))
        torch.save(decD.state_dict(), './{0}/decD.pth'.format(opt.save_dir))
        
        torch.save(optEnc.state_dict(), './{0}/optEnc.pth'.format(opt.save_dir))
        torch.save(optDec.state_dict(), './{0}/optDec.pth'.format(opt.save_dir))
        torch.save(optEncD.state_dict(), './{0}/optEncD.pth'.format(opt.save_dir))
        torch.save(optDecD.state_dict(), './{0}/optDecD.pth'.format(opt.save_dir))
        
        
        pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))
        
#     optEnc.param_groups[0]['lr'] = learningRate*(0.999**epoch)
#     optDec.param_groups[0]['lr'] = learningRate*(0.999**epoch)
#     optEncD.param_groups[0]['lr'] = learningRate*(0.999**epoch)
                  
                  
print('Finished Training')
