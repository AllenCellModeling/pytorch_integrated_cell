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
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--myseed', type=int, default=0, help='random seed')
parser.add_argument('--nlatentdim', type=int, default=32, help='number of latent dimensions')
parser.add_argument('--lrEnc', type=float, default=0.0005, help='learning rate for encoder')
parser.add_argument('--lrDec', type=float, default=0.0005, help='learning rate for decoder')
parser.add_argument('--lrEncD', type=float, default=0.00005, help='learning rate for encD')
parser.add_argument('--lrDecD', type=float, default=0.00005, help='learning rate for decD')
parser.add_argument('--encDRatio', type=float, default=1, help='scalar applied to the update gradient from encD')
parser.add_argument('--decDRatio', type=float, default=1E-4, help='scalar applied to the update gradient from decD')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--nepochs', type=int, default=250, help='total number of epochs')
parser.add_argument('--clamp_lower', type=float, default=-0.01, help='lower clamp for wasserstein gan')
parser.add_argument('--clamp_upper', type=float, default=0.01, help='upper clamp for wasserstein gan')
parser.add_argument('--save_dir', default='./waaegan/', help='save dir')
parser.add_argument('--saveProgressIter', type=int, default=1, help='number of iterations between saving progress')
parser.add_argument('--saveStateIter', type=int, default=5, help='number of iterations between saving progress')
parser.add_argument('--imsize', type=int, default=128, help='pixel size of images used')   
parser.add_argument('--imdir', default='/root/images/release_4_1_17_2D', help='location of images')
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

#     print(img.shape[0]*10)
#     print(img.shape[1]*10)    
#     fig = plt.figure(figsize = [img.shape[0]/2, img.shape[1]/2])
    

#     ax = fig.add_subplot(111)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.imshow(img)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
    
    
enc = waaegan.Enc(opt.nlatentdim, opt.imsize)
dec = waaegan.Dec(opt.nlatentdim, opt.imsize)
encD = waaegan.EncD(opt.nlatentdim)
decD = waaegan.DecD(1, opt.imsize)

enc.apply(weights_init)
dec.apply(weights_init)
encD.apply(weights_init)
decD.apply(weights_init)

gpu_id = opt.gpu_id
nlatentdim = opt.nlatentdim

enc.cuda(gpu_id)
dec.cuda(gpu_id)
encD.cuda(gpu_id)
decD.cuda(gpu_id)

criterion = nn.BCELoss()

# optEnc = optim.RMSprop(enc.parameters(), lr=opt.lrEnc)
# optDec = optim.RMSprop(dec.parameters(), lr=opt.lrDec)
# optEncD = optim.RMSprop(encD.parameters(), lr=opt.lrEncD)
# optDecD = optim.RMSprop(decD.parameters(), lr=opt.lrDecD)

optEnc = optim.Adam(enc.parameters(), lr=opt.lrEnc, betas=(0.5, 0.999))
optDec = optim.Adam(dec.parameters(), lr=opt.lrDec, betas=(0.5, 0.999))
optEncD = optim.Adam(encD.parameters(), lr=opt.lrEncD, betas=(0.5, 0.999))
optDecD = optim.Adam(decD.parameters(), lr=opt.lrDecD, betas=(0.5, 0.999))

ndat = dp.get_n_train()
ndat = 1000

logger = SimpleLogger.SimpleLogger(('epoch', 'iter', 'reconLoss', 'minimaxEncDLoss', 'encDLoss', 'minimaxDecDLoss', 'decDLoss', 'time'), '[%d][%d] reconLoss: %.6f mmEncD: %.6f encD: %.6f mmDecD: %.6f decD: %.6f time: %.2f')

one = torch.FloatTensor([1]).cuda(gpu_id)
mone = one * -1

gen_iterations = 0
iteration = 0
for epoch in range(1, opt.nepochs+1): # loop over the dataset multiple times

    
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
        if gen_iterations <= 5 or iteration % 25 == 0:
            Diters = opt.DitersAlt
        else:
            Diters = opt.Diters
        j = 0

        rand_inds_encD = np.random.permutation(ndat)
        niter = len(range(0, len(rand_inds_encD), opt.batch_size))
        inds_encD = (rand_inds_encD[i:i+opt.batch_size] for i in range(0, len(rand_inds_encD), opt.batch_size))
        
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
            zReal = Variable(torch.Tensor(batsize, nlatentdim).uniform_(-2, 2)).cuda(gpu_id)
            
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
            
            
            xHat = dec(zFake)
            
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
        
        zReal = Variable(torch.Tensor(batsize, nlatentdim).uniform_(-2, 2)).cuda(gpu_id)
        xHat = dec(zReal)
        
        minimaxDecDLoss2 = decD(xHat)
        minimaxDecDLoss2.backward(one*opt.decDRatio, retain_variables=True)
        
        optDec.step()

        zAll.append(zFake.data)
        
        stop = time.time()
        deltaT = stop-start
        
        logger.add((epoch, iteration, reconLoss.data[0], minimaxEncDLoss.data[0][0], encDLoss.data[0][0], minimaxDecDLoss.data[0], decDLoss.data[0], deltaT))

    gen_iterations += 1
    
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

        plt.gca().cla() 
        plt.scatter(zAll[:,0], zAll[:,1])
        plt.xlim([-4, 4]) 
        plt.ylim([-4, 4])     
        plt.axis('equal')
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.title('latent space embedding')
        plt.savefig('./{0}/embedding_{1}.png'.format(opt.save_dir, epoch), dpi=75)
        
        pickle.dump(zAll, open('./{0}/embedding.pkl'.format(opt.save_dir), 'wb'))
        pickle.dump(logger, open('./{0}/logger.pkl'.format(opt.save_dir), 'wb'))

    if (epoch % opt.saveStateIter) == 0:
#         for saving and loading see:
#         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718
        torch.save(enc.state_dict(), './{0}/enc.pth'.format(opt.save_dir))
        torch.save(dec.state_dict(), './{0}/dec.pth'.format(opt.save_dir))
        torch.save(encD.state_dict(), './{0}/encD.pth'.format(opt.save_dir))
        
        torch.save(optEnc.state_dict(), './{0}/optEnc.pth'.format(opt.save_dir))
        torch.save(optDec.state_dict(), './{0}/optDec.pth'.format(opt.save_dir))
        torch.save(optEncD.state_dict(), './{0}/optEncD.pth'.format(opt.save_dir))
        
        pickle.dump(opt, open('./{0}/opt.pkl'.format(opt.save_dir), 'wb'))
            
        
#     optEnc.param_groups[0]['lr'] = learningRate*(0.999**epoch)
#     optDec.param_groups[0]['lr'] = learningRate*(0.999**epoch)
#     optEncD.param_groups[0]['lr'] = learningRate*(0.999**epoch)
                  
                  
print('Finished Training')
