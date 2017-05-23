import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

def iteration(enc, dec, encD, decD, 
              optEnc, optDec, optEncD, optDecD, 
              critRecon, critZClass, critZRef, critEncD, critDecD,
              dataProvider, opt):
    
    gpu_id = opt.gpu_ids[0]
    
    one = torch.FloatTensor([1]).cuda(gpu_id)
    mone = one * -1
    
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

    if opt.iter <= 100 or (opt.iter % 25) == 0:
        Diters = opt.DitersAlt
    else:
        Diters = opt.Diters
    
    for j in range(0, Diters):
        # clamp parameters to a cube
        for p in encD.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        for p in decD.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        rand_inds_encD = np.random.permutation(opt.ndat)
        inds = rand_inds_encD[0:opt.batch_size]

        x = Variable(dataProvider.get_images(inds,'train')).cuda(gpu_id)

        zAll = enc(x)
        zFake = zAll[-1]
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
        
        encDLoss = (errEncD_real - errEncD_fake)
        optEncD.step()
        
        xHat = dec(zAll)

        errDecD_real = decD(x)
        errDecD_real.backward(one, retain_variables=True)

        errDecD_fake = decD(xHat)
        errDecD_fake.backward(mone, retain_variables=True)
                
        zAll[-1] = zReal

        errDecD_fake2 = decD(dec(zAll))
        errDecD_fake2.backward(mone, retain_variables=True)

        decDLoss = errDecD_real - (errDecD_fake + errDecD_fake2)/2
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

    
    if opt.nClasses > 0:
        classes = Variable(dataProvider.get_classes(inds,'train')).cuda(gpu_id)

    if opt.nRef > 0:
        ref = Variable(dataProvider.get_ref(inds,'train')).cuda(gpu_id)
    
    ## train the autoencoder
    zAll = enc(x)
    xHat = dec(zAll)    
    
    c = 0        
    if opt.nClasses > 0:
        classLoss = critZClass(zAll[c], classes)
        classLoss.backward(retain_variables=True)        
        c += 1
        
    if opt.nRef > 0:
        refLoss = critZRef(zAll[c], ref)
        refLoss.backward(retain_variables=True)        
        c += 1
        
    reconLoss = critRecon(xHat, x)
    reconLoss.backward(retain_variables=True)        

    #update wrt encD
    minimaxEncDLoss = encD(zAll[c])
    (minimaxEncDLoss.mul(opt.encDRatio)).backward(retain_variables=True)
    
    optEnc.step()
    
    for p in enc.parameters():
        p.requires_grad = False
    
    #update wrt decD(dec(enc(X)))
    minimaxDecDLoss = decD(xHat)
    minimaxDecDLoss.backward(one*opt.decDRatio, retain_variables=True)
    
    #update wrt decD(dec(Z))
    zAll[c] = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)

    minimaxDecDLoss2 = decD(dec(zAll))
    minimaxDecDLoss2.backward(one*opt.decDRatio, retain_variables=True)
    
    optDec.step()
    minimaxDecDLoss = (minimaxDecDLoss+minimaxDecDLoss2)/2
    
    errors = (reconLoss.data[0],)
    if opt.nClasses > 0:
        errors += (classLoss.data[0],)
    
    if opt.nRef > 0:
        errors += (refLoss.data[0],)
    
    errors += (minimaxEncDLoss.data[0], encDLoss.data[0], minimaxDecDLoss.data[0].cpu()[0], decDLoss.data[0].cpu()[0])
    
    return errors, zFake.data
    