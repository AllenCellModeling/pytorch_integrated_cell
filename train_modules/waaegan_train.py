import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

def improved_penalty(interp_alpha, xreal, xfake, netD, opt):
    interp_alpha.data.uniform_()
    
    interp_points = Variable((interp_alpha.expand_as(xreal)*xreal+(1-interp_alpha.expand_as(xreal))*xfake).data, requires_grad=True)
    errD_interp_vec = netD(interp_points)
    errD_gradient, = torch.autograd.grad(errD_interp_vec.sum(), interp_points, create_graph=True)
    lip_est = (errD_gradient**2).view(opt.batch_size,-1).sum(1)
    lip_loss = opt.improved_penalty*((1.0-lip_est)**2).mean(0).view(1)
    lip_loss.backward()
    
def iteration(enc, dec, encD, decD, 
              optEnc, optDec, optEncD, optDecD, 
              critRecon, critZClass, critZRef, critEncD, critDecD,
              dataProvider, opt):
    
    gpu_id = opt.gpu_ids[0]
    
    one = torch.FloatTensor([1]).cuda(gpu_id)
    mone = one * -1
    
    if opt.improved:
        interp_alpha = torch.FloatTensor(opt.batch_size, 1).cuda(gpu_id)
        interp_alpha = Variable(interp_alpha)
    
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
        if not opt.improved:
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
        errEncD_real_vec = encD(zReal)
        errEncD_real = torch.mean(errEncD_real_vec)
        errEncD_real.backward(one, retain_graph=True)

        # train with fake
        errEncD_fake_vec = encD(zFake)
        errEncD_fake = torch.mean(errEncD_fake_vec)
        errEncD_fake.backward(mone, retain_graph=True)
        
        encDLoss = (errEncD_real - errEncD_fake)
        
        if opt.improved:
            improved_penalty(interp_alpha, zReal, zFake, encD, opt)
            
        
        optEncD.step()
        
        xHat = dec(zAll)

        errDecD_real_vec = decD(x)
        errDecD_real = torch.mean(errDecD_real_vec)
        errDecD_real.backward(one, retain_graph=True)

        errDecD_fake_vec = decD(xHat)
        errDecD_fake = torch.mean(errDecD_fake_vec)
        errDecD_fake.backward(mone, retain_graph=True)
            
        if opt.improved:
            improved_penalty(interp_alpha, x, xHat, decD, opt)
            
        zAll[-1] = zReal

        xHat2 = dec(zAll)
        errDecD_fake2_vec = decD(xHat2)
        errDecD_fake2 = torch.mean(errDecD_fake2_vec)
        errDecD_fake2.backward(mone, retain_graph=True)

        decDLoss = errDecD_real - (errDecD_fake + errDecD_fake2)/2
        
        if opt.improved:
            improved_penalty(interp_alpha, x, xHat2, decD, opt)
        
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
        classLoss.backward(retain_graph=True)        
        c += 1
        
    if opt.nRef > 0:
        refLoss = critZRef(zAll[c], ref)
        refLoss.backward(retain_graph=True)        
        c += 1
        
    reconLoss = critRecon(xHat, x)
    reconLoss.backward(retain_graph=True)        

    #update wrt encD
    minimaxEncDLoss_vec = encD(zAll[c])
    minimaxEncDLoss = torch.mean(minimaxEncDLoss_vec)
    minimaxEncDLoss.backward(one*opt.encDRatio, retain_graph=True)
    
    optEnc.step()
    
    for p in enc.parameters():
        p.requires_grad = False
    
    #update wrt decD(dec(enc(X)))
    minimaxDecDLoss_vec = decD(xHat)
    minimaxDecDLoss = torch.mean(minimaxDecDLoss_vec)
    minimaxDecDLoss.backward(one*opt.decDRatio, retain_graph=True)
    
    #update wrt decD(dec(Z))
    zAll[c] = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim)).cuda(gpu_id)

    minimaxDecDLoss2_vec = decD(dec(zAll))
    minimaxDecDLoss2 = torch.mean(minimaxDecDLoss2_vec)
    minimaxDecDLoss2.backward(one*opt.decDRatio, retain_graph=True)
    
    optDec.step()
    minimaxDecDLoss = (minimaxDecDLoss+minimaxDecDLoss2)/2
    
    errors = (reconLoss.data[0],)
    if opt.nClasses > 0:
        errors += (classLoss.data[0],)
    
    if opt.nRef > 0:
        errors += (refLoss.data[0],)
    
    errors += (minimaxEncDLoss.data[0], encDLoss.data[0], minimaxDecDLoss.data[0], decDLoss.data[0])
    
    return errors, zFake.data
    