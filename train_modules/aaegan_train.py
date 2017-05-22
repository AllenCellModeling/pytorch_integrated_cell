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

    inds = next(inds_encD)
    x = Variable(dataProvider.get_images(inds,'train')).cuda(gpu_id)
    
    if opt.nClasses > 0:
        classes = Variable(dataProvider.get_classes(inds,'train')).cuda(gpu_id)
    
    if opt.nRef > 0:
        ref = Variable(dataProvider.get_ref(inds,'train')).cuda(gpu_id)
    
    zAll = enc(x)
    
    for var in zAll:
        var.detach_()
    
    xHat = dec(zAll)
  
    zReal = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)
    zFake = zAll[-1]

    optEnc.zero_grad()
    optDec.zero_grad()
    optEncD.zero_grad()
    optDecD.zero_grad()

    ### train encD
    y_zReal = Variable(torch.ones(opt.batch_size)).cuda(gpu_id)
    y_zFake = Variable(torch.zeros(opt.batch_size)).cuda(gpu_id)
    
    # train with real
    yHat_zReal = encD(zReal)
    errEncD_real = critEncD(yHat_zReal, y_zReal)
    errEncD_real.backward(retain_variables=True)

    # train with fake
    yHat_zFake = encD(zFake)
    errEncD_fake = critEncD(yHat_zFake, y_zFake)
    errEncD_fake.backward(retain_variables=True)
    
    encDLoss = (errEncD_real + errEncD_fake)/2

    ###Train decD 
    if opt.nClasses > 0:
        y_xReal = classes
        y_xFake = Variable(torch.LongTensor(opt.batch_size).fill_(opt.nClasses)).cuda(gpu_id)
    else:
        y_xReal = Variable(torch.ones(opt.batch_size)).cuda(gpu_id)
        y_xFake = Variable(torch.zeros(opt.batch_size)).cuda(gpu_id)
    
    yHat_xReal = decD(x)
    errDecD_real = critDecD(yHat_xReal, y_xReal)
    # errDecD_real.backward(retain_variables=True)

    #train with fake, reconstructed
    yHat_xFake = decD(xHat)        
    errDecD_fake = critDecD(yHat_xFake, y_xFake)
    # errDecD_fake.backward(retain_variables=True)
    
    #train with fake, sampled and decoded
    zAll[-1] = zReal
    
    yHat_xFake2 = decD(dec(zAll))
    errEncD_fake2 = critDecD(yHat_xFake2, y_xFake)
    # errEncD_fake2.backward(retain_variables=True)

    decDLoss = (errDecD_real + (errDecD_fake + errEncD_fake2)/2)/2
    decDLoss.backward(retain_variables=True)
    optDecD.step()z

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
    yHatFake = encD(zAll[c])
    minimaxEncDLoss = critEncD(yHatFake, y_zReal)
    (minimaxEncDLoss.mul(opt.encDRatio)).backward(retain_variables=True)

    optEnc.step()
    
    for p in enc.parameters():
        p.requires_grad = False
    
    #update wrt decD(dec(enc(X)))
    yHat_xFake = decD(xHat)
    minimaxDecDLoss = critDecD(yHat_xFake, y_xReal)
    
    #update wrt decD(dec(Z))
    zAll[c] = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)
    xHat = dec(zAll)

    yHat_xFake2 = decD(xHat)
    minimaxDecDLoss2 = critDecD(yHat_xFake2, y_xReal)
    
    minimaxDecLoss = (minimaxDecDLoss+minimaxDecDLoss2)/2
    (minimaxDecLoss.mul(opt.decDRatio)).backward(retain_variables=True)
    
    optDec.step()

    
    errors = (reconLoss.data[0],)
    if opt.nClasses > 0:
        errors += (classLoss.data[0],)
    
    if opt.nRef > 0:
        errors += (refLoss.data[0],)
    
    errors += (minimaxEncDLoss.data[0], encDLoss.data[0], minimaxDecLoss.data[0], decDLoss.data[0])
    
    return errors, zFake.data
    