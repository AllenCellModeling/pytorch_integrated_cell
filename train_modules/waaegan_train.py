import importlib
import torch
import SimpleLogger
import torch.optim as optim
import os
import pickle
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from model_utils import set_gpu_recursive


def iteration(models, optimizers, criterions, dataProvider, iteration, opt):
    gpu_id = opt.gpu_ids[0]
    
    
    enc = models[0]
    dec = models[1]
    encD = models[2]
    decD = models[3]
    
    optEnc = optimizers[0]
    optDec = optimizers[1]
    optEncD = optimizers[2]
    optDecD = optimizers[3]
    
    criterion = criterions[0]

    one = torch.FloatTensor([1]).cuda(gpu_id)
    mone = one * -1


    yReal = Variable(torch.ones(opt.batch_size)).cuda(gpu_id)
    yFake = Variable(torch.zeros(opt.batch_size)).cuda(gpu_id)

    ###update the discriminator
    #maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))
    for p in encD.parameters(): # reset requires_grad
        p.requires_grad = True # they are set to False below in netG update

    for p in decD.parameters():
        p.requires_grad = True

    for p in enc.parameters():
        p.requires_grad = False

    for p in dec.parameters():
        p.requires_grad = False

    # train the discriminator Diters times        
    if iteration <= 100 or (iteration % 25) == 0:
        Diters = opt.DitersAlt
    else:
        Diters = opt.Diters

    rand_inds_encD = np.random.permutation(opt.ndat)
    niter = len(range(0, len(rand_inds_encD), opt.batch_size))
    inds_encD = (rand_inds_encD[i:i+opt.batch_size] for i in range(0, len(rand_inds_encD), opt.batch_size))

    # pdb.set_trace()
    j = 0
    while j < Diters and j < niter:
        j += 1

        # clamp parameters to a cube
        for p in encD.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        for p in decD.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        x = Variable(dataProvider.get_images(next(inds_encD),'train')).cuda(gpu_id)

        zFake = enc(x)
        #pick a distribution that is obvious when you plot it
        # zReal = Variable(torch.Tensor(batsize, nlatentdim).uniform_(-2, 2)).cuda(gpu_id)
        zReal = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)

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

    for p in enc.parameters():
        p.requires_grad = True

    for p in dec.parameters():
        p.requires_grad = True  

    for p in encD.parameters():
        p.requires_grad = False

    for p in decD.parameters():
        p.requires_grad = False

    optEnc.zero_grad()
    optDec.zero_grad()
    optEncD.zero_grad()  
    optDecD.zero_grad()          

    zFake = enc(x)
    xHat = dec(zFake)
    reconLoss = criterion(xHat, x)
    reconLoss.backward(retain_variables=True)

    minimaxEncDLoss = encD(zFake)
    minimaxEncDLoss.backward(one*opt.encDRatio, retain_variables=True)

    optEnc.step()

    # optDec.step()
    # optDec.zero_grad()

    for p in enc.parameters():
        p.requires_grad = False

    minimaxDecDLoss = decD(xHat)
    minimaxDecDLoss.backward(one*opt.decDRatio, retain_variables=True)

    zReal = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)
    xHat = dec(zReal.detach())

    minimaxDecDLoss2 = decD(xHat)
    minimaxDecDLoss2.backward(one*opt.decDRatio, retain_variables=True)

    optDec.step()

    # pdb.set_trace()
    errors = (reconLoss.data[0], minimaxEncDLoss.data[0], encDLoss.data[0], minimaxDecDLoss.data[0], decDLoss.data[0])
    
    return errors, zFake.data
    