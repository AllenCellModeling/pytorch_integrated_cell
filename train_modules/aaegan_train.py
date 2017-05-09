import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

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

    rand_inds_encD = np.random.permutation(opt.ndat)
    niter = len(range(0, len(rand_inds_encD), opt.batch_size))
    inds_encD = (rand_inds_encD[i:i+opt.batch_size] for i in range(0, len(rand_inds_encD), opt.batch_size))

    x = Variable(dataProvider.get_images(next(inds_encD),'train')).cuda(gpu_id)
    zFake = enc(x)
    xHat = dec(zFake.detach())
  
    zReal = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)

    optEnc.zero_grad()
    optDec.zero_grad()
    optEncD.zero_grad()
    optDecD.zero_grad()

    ### train encD
    # train with real
    yHatReal = encD(zReal)
    errEncD_real = criterion(yHatReal, yReal)
    errEncD_real.backward(retain_variables=True)

    # train with fake
    yHatFake = encD(zFake)
    errEncD_fake = criterion(yHatFake, yFake)
    errEncD_fake.backward(retain_variables=True)
    encDLoss = errEncD_real + errEncD_fake

    optEncD.step()

    #Train decD 
    yHatReal = decD(x)
    errDecD_real = criterion(yHatReal, yReal)
    errDecD_real.backward(retain_variables=True)

    yHatFake = decD(xHat)
    errDecD_fake = criterion(yHatFake, yFake)
    errDecD_fake.backward(retain_variables=True)

    decDLoss = errDecD_real + errDecD_fake
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

    ## train the autoencoder
    zFake = enc(x)
    xHat = dec(zFake)
    reconLoss = criterion(xHat, x)
    reconLoss.backward(retain_variables=True)

    #update wrt encD
    yHatFake = encD(zFake)
    minimaxEncDLoss = criterion(yHatFake, yReal)
    (minimaxEncDLoss.mul(opt.encDRatio)).backward(retain_variables=True)

    optEnc.step()
    
    for p in enc.parameters():
        p.requires_grad = False
    
#     #this is optionsl
#     optDec.step()
#     optDec.zero_grad()
#     xHat = dec(zFake.detach())
    


    #update wrt decD(dec(enc(X)))
    yHatFake = decD(xHat)
    minimaxDecDLoss = criterion(yHatFake, yReal)
    (minimaxDecDLoss.mul(opt.decDRatio)).backward(retain_variables=True)
    
    #update wrt decD(dec(Z))
    zReal = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)
    xHat = dec(zReal)

    yHatFake = decD(xHat)
    minimaxDecDLoss2 = criterion(yHatFake, yReal)
    (minimaxDecDLoss2.mul(opt.decDRatio)).backward(retain_variables=True)

    optDec.step()

    # pdb.set_trace()
    errors = (reconLoss.data[0], minimaxEncDLoss.data[0], encDLoss.data[0], minimaxDecDLoss.data[0], decDLoss.data[0])
    
    return errors, zFake.data
    