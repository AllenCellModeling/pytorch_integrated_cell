import torch
import importlib
import torch.optim as optim
import SimpleLogger
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.misc
import pickle

def init_opts(opt, opt_default):
    vars_default = vars(opt_default)
    for var in vars_default:
        if not hasattr(opt, var):
            setattr(opt, var, getattr(opt_default, var))
    return opt

def tensor2img(img, opt):
    
    imresize = list(img.size())
    imresize[1] = 3
    
    img_out = torch.zeros(tuple(imresize))
    
    img = img.numpy()
    # img += 1
    # img /= 2
    
    img_tmp = np.zeros(imresize)
    img_tmp[:, opt.channelInds] = img
    img = img_tmp
    
    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    img = np.transpose(img, [0,2,3,1])
    img = np.concatenate(img[:], 1)
    
    return img

def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                if gpu_id != -1:
                    var[key] = var[key].cuda(gpu_id)
                else:
                    var[key] = var[key].cpu()
            except:
                pass
    return var  

def sampleUniform (batsize, nlatentdim): 
    return torch.Tensor(batsize, nlatentdim).uniform_(-1, 1)

def sampleGaussian (batsize, nlatentdim): 
    return torch.Tensor(batsize, nlatentdim).normal_()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    

def load_model(model_provider, opt):
    model = importlib.import_module("models." + opt.model_name)
 
    enc = model_provider.Enc(opt.nlatentdim, opt.imsize, opt.nch, opt.gpu_ids, opt)
    dec = model_provider.Dec(opt.nlatentdim, opt.imsize, opt.nch, opt.gpu_ids, opt)
    encD = model_provider.EncD(opt.nlatentdim, opt.gpu_ids, opt)
    decD = model_provider.DecD(1, opt.imsize, opt.nch, opt.gpu_ids, opt)

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

    if opt.optimizer == 'RMSprop':
        optEnc = optim.RMSprop(enc.parameters(), lr=opt.lrEnc)
        optDec = optim.RMSprop(dec.parameters(), lr=opt.lrDec)
        optEncD = optim.RMSprop(encD.parameters(), lr=opt.lrEncD)
        optDecD = optim.RMSprop(decD.parameters(), lr=opt.lrDecD)
    elif opt.optimizer == 'adam':
        optEnc = optim.Adam(enc.parameters(), lr=opt.lrEnc, betas=(0.5, 0.999))
        optDec = optim.Adam(dec.parameters(), lr=opt.lrDec, betas=(0.5, 0.999))
        optEncD = optim.Adam(encD.parameters(), lr=opt.lrEncD, betas=(0.5, 0.999))
        optDecD = optim.Adam(decD.parameters(), lr=opt.lrDecD, betas=(0.5, 0.999))
    
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
        from model_utils import sampleUniform as latentSample
        
    elif opt.latentDistribution == 'gaussian':
        from model_utils import sampleGaussian as latentSample
        

    opt.latentSample = latentSample   
    
    return models, optimizers, criterions, logger, opt


def save_progress(models, dataProvider, logger, zAll, epoch, opt):
    enc = models[0]
    dec = models[1]
    
    gpu_id = opt.gpu_ids[0]
    
    enc.train(False)
    dec.train(False)

    x = Variable(dataProvider.get_images(np.arange(0,10),'train')).cuda(gpu_id)
    xHat = dec(enc(x))
    imgX = tensor2img(x.data.cpu(), opt)
    imgXHat = tensor2img(xHat.data.cpu(), opt)
    imgTrainOut = np.concatenate((imgX, imgXHat), 0)

    x = Variable(dataProvider.get_images(np.arange(0,10),'test')).cuda(gpu_id)
    xHat = dec(enc(x))
    imgX = tensor2img(x.data.cpu(), opt)
    imgXHat = tensor2img(xHat.data.cpu(), opt)
    imgTestOut = np.concatenate((imgX, imgXHat), 0)

    imgOut = np.concatenate((imgTrainOut, imgTestOut))

    scipy.misc.imsave('./{0}/progress_{1}.png'.format(opt.save_dir, epoch), imgOut)

    enc.train(True)
    dec.train(True)

    # pdb.set_trace()
    # zAll = torch.cat(zAll,0).cpu().numpy()

    pickle.dump(zAll, open('./{0}/embedding_tmp.pkl'.format(opt.save_dir), 'wb'))
    pickle.dump(logger, open('./{0}/logger_tmp.pkl'.format(opt.save_dir), 'wb'))
    
def save_state(models, optimizers, logger, zAll, opt):
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
   