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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    

      
        
def load(model_provider, opt):
    model = importlib.import_module("models." + opt.model_name)
 
    enc = model_provider.Enc(opt.nlatentdim, opt.imsize, opt.gpu_ids)
    dec = model_provider.Dec(opt.nlatentdim, opt.imsize, opt.gpu_ids)
    encD = model_provider.EncD(opt.nlatentdim, opt.gpu_ids)
    decD = model_provider.DecD(1, opt.imsize, opt.gpu_ids)

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

    # optEnc = optim.RMSprop(enc.parameters(), lr=opt.lrEnc)
    # optDec = optim.RMSprop(dec.parameters(), lr=opt.lrDec)
    # optEncD = optim.RMSprop(encD.parameters(), lr=opt.lrEncD)
    # optDecD = optim.RMSprop(decD.parameters(), lr=opt.lrDecD)

    optEnc = optim.Adam(enc.parameters(), lr=opt.lrEnc, betas=(0.5, 0.9))
    optDec = optim.Adam(dec.parameters(), lr=opt.lrDec, betas=(0.5, 0.9))
    optEncD = optim.Adam(encD.parameters(), lr=opt.lrEncD, betas=(0.5, 0.9))
    optDecD = optim.Adam(decD.parameters(), lr=opt.lrDecD, betas=(0.5, 0.9))
    
    
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

        optEnc.state = set_gpu_recursive(optEnc.state, gpu_id)
        optDec.state = set_gpu_recursive(optDec.state, gpu_id)
        optEncD.state = set_gpu_recursive(optEncD.state, gpu_id)
        optDecD.state = set_gpu_recursive(optDecD.state, gpu_id)

        enc.cuda(gpu_id)
        dec.cuda(gpu_id)
        encD.cuda(gpu_id)
        decD.cuda(gpu_id)                           

        # opt = pickle.load(open( '{0}/opt.pkl'.format(opt.save_dir), "rb" ))
        logger = pickle.load(open( '{0}/logger.pkl'.format(opt.save_dir), "rb" ))

        this_epoch = max(logger.log['epoch']) + 1
        iteration = max(logger.log['iter'])

    
    models = (enc, dec, encD, decD)
    optimizers = (optEnc, optDec, optEncD, optDecD)
    criterions = ([nn.BCELoss()])

    if opt.latentDistribution == 'uniform':
        def latentSample (batsize, nlatentdim): return torch.Tensor(opt.batch_size, nlatentdim).uniform_(-1, 1)
    elif opt.latentDistribution == 'gaussian':
        def latentSample (batsize, nlatentdim): return torch.Tensor(opt.batch_size, nlatentdim).normal_()

    opt.latentSample = latentSample   
    
    return models, optimizers, criterions, logger, opt

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
    