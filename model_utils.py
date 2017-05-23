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

import matplotlib.pyplot as plt
from imgToProjection import imgtoprojection

import pdb

def init_opts(opt, opt_default):
    vars_default = vars(opt_default)
    for var in vars_default:
        if not hasattr(opt, var):
            setattr(opt, var, getattr(opt_default, var))
    return opt

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

def tensor2img(img):
    
    img = img.numpy()
    im_out = list()
    for i in range(0, img.shape[0]):
        im_out.append(img[i])

    img = np.concatenate(im_out, 2)

    if len(img.shape) == 3:
        img = np.expand_dims(img, 3)

    colormap = 'hsv'

    colors = plt.get_cmap(colormap)(np.linspace(0, 1, img.shape[0]+1))

    # img = np.swapaxes(img, 2,3)
    img = imgtoprojection(np.swapaxes(img, 1, 3), colors = colors, global_adjust=True)
    img = np.swapaxes(img, 0, 2)

    return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
        
def get_latent_embeddings(enc, dp, opt):
    enc.eval()
    gpu_id = opt.gpu_ids[0]
    
    modes = ('test', 'train')
    
    embedding = dict()
    
    for mode in modes:
        ndat = dp.get_n_dat(mode)
        embeddings = torch.zeros(ndat, opt.nlatentdim)
        
        inds = list(range(0,ndat))
        data_iter = [inds[i:i+opt.batch_size] for i in range(0, len(inds), opt.batch_size)]

        for i in range(0, len(data_iter)):
            x = Variable(dp.get_images(data_iter[i], mode).cuda(gpu_id))
            zAll = enc(x)
            
            embeddings.index_copy_(0, torch.LongTensor(data_iter[i]), zAll[-1].data[:].cpu())
        
        embedding[mode] = embeddings
        
    return embedding
        
def load_model(model_provider, opt):
    model = importlib.import_module("models." + opt.model_name)
 
    enc = model_provider.Enc(opt.nlatentdim, opt.nClasses, opt.nRef, opt.imsize, opt.nch, opt.gpu_ids, opt)
    dec = model_provider.Dec(opt.nlatentdim, opt.nClasses, opt.nRef, opt.imsize, opt.nch, opt.gpu_ids, opt)
    encD = model_provider.EncD(opt.nlatentdim, opt.gpu_ids, opt)
    decD = model_provider.DecD(opt.nClasses+1, opt.imsize, opt.nch, opt.gpu_ids, opt)

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
    
    
    columns = ('epoch', 'iter', 'reconLoss',)
    print_str = '[%d][%d] reconLoss: %.6f'
    
    if opt.nClasses > 0:
        columns += ('classLoss',)
        print_str += ' classLoss: %.6f'
        
    if opt.nRef > 0:
        columns += ('refLoss',)
        print_str += ' refLoss: %.6f'
    
    columns += ('minimaxEncDLoss', 'encDLoss', 'minimaxDecDLoss', 'decDLoss', 'time')
    print_str += ' mmEncD: %.6f encD: %.6f mmDecD: %.6f decD: %.6f time: %.2f'
    
    logger = SimpleLogger.SimpleLogger(columns,  print_str)

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

    models = {'enc': enc, 'dec': dec, 'encD': encD, 'decD': decD}
    
    optimizers = dict()
    optimizers['optEnc'] = optEnc
    optimizers['optDec'] = optDec
    optimizers['optEncD'] = optEncD
    optimizers['optDecD'] = optDecD
    
    criterions = dict()
    criterions['critRecon'] = nn.BCELoss()
    criterions['critZClass'] = nn.NLLLoss()    
    criterions['critZRef'] = nn.MSELoss()
    criterions['critEncD'] = nn.BCELoss()
    
    if opt.nClasses > 0:
        criterions['critDecD'] = nn.NLLLoss()
    else:
        criterions['critDecD'] = nn.BCELoss()
 
    if opt.latentDistribution == 'uniform':
        from model_utils import sampleUniform as latentSample
        
    elif opt.latentDistribution == 'gaussian':
        from model_utils import sampleGaussian as latentSample
        

    opt.latentSample = latentSample   
    
    return models, optimizers, criterions, logger, opt

def maybe_save(epoch, epoch_next, models, optimizers, logger, zAll, dp, opt):
    saved = False
    if epoch != epoch_next and ((epoch_next % opt.saveProgressIter) == 0 or (epoch_next % opt.saveStateIter) == 0):

        zAll = torch.cat(zAll,0).cpu().numpy()

        if (epoch_next % opt.saveProgressIter) == 0:
            print('saving progress')
            save_progress(models['enc'], models['dec'], dp, logger, zAll, opt)

        if (epoch_next % opt.saveStateIter) == 0:
            print('saving state')
            save_state(**models, **optimizers, logger=logger, zAll=zAll, opt=opt)

        saved = True
        
    return saved
            

def save_progress(enc, dec, dataProvider, logger, embedding, opt):
    
    gpu_id = opt.gpu_ids[0]
    
    epoch = max(logger.log['epoch'])
    
    enc.train(False)
    dec.train(False)

    x = Variable(dataProvider.get_images(np.arange(0,10),'train')).cuda(gpu_id)
    xHat = dec(enc(x))
    imgX = tensor2img(x.data.cpu())
    imgXHat = tensor2img(xHat.data.cpu())
    imgTrainOut = np.concatenate((imgX, imgXHat), 0)

    x = Variable(dataProvider.get_images(np.arange(0,10),'test')).cuda(gpu_id)
    xHat = dec(enc(x))
    imgX = tensor2img(x.data.cpu())
    imgXHat = tensor2img(xHat.data.cpu())
    imgTestOut = np.concatenate((imgX, imgXHat), 0)

    imgOut = np.concatenate((imgTrainOut, imgTestOut))

    scipy.misc.imsave('./{0}/progress_{1}.png'.format(opt.save_dir, int(epoch)), imgOut)

    enc.train(True)
    dec.train(True)

    # pdb.set_trace()
    # zAll = torch.cat(zAll,0).cpu().numpy()

    pickle.dump(embedding, open('./{0}/embedding_tmp.pkl'.format(opt.save_dir), 'wb'))
    pickle.dump(logger, open('./{0}/logger_tmp.pkl'.format(opt.save_dir), 'wb'))
    
    

    ### History
    plt.figure()

    for i in range(2, len(logger.fields)-1):
        field = logger.fields[i]
        plt.plot(logger.log['iter'], logger.log[field], label=field)

    plt.legend()
    plt.title('History')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('{0}/history.png'.format(opt.save_dir), bbox_inches='tight')
    plt.close()

    ### Short History
    history = 10000
    
    if len(logger.log['epoch']) < history:
        history = int(len(logger.log['epoch'])/2)
    
    ydat = [logger.log['encDLoss'], logger.log['decDLoss'], logger.log['minimaxEncDLoss'], logger.log['minimaxDecDLoss']]
    ymin = np.min(ydat.append(logger.log['reconLoss']))
    ymax = np.max(ydat)
    plt.ylim([ymin, ymax])

    x = logger.log['iter'][-history:]
    y = logger.log['reconLoss'][-history:]

    epochs = np.floor(np.array(logger.log['epoch'][-history:-1]))
    losses = np.array(logger.log['reconLoss'][-history:-1])
    iters = np.array(logger.log['iter'][-history:-1])
    uepochs = np.unique(epochs)

    epoch_losses = np.zeros(len(uepochs))
    epoch_iters = np.zeros(len(uepochs))
    i = 0
    for uepoch in uepochs:
        inds = np.equal(epochs, uepoch)
        loss = np.mean(losses[inds])
        epoch_losses[i] = loss
        epoch_iters[i] = np.mean(iters[inds])
        i+=1

    mval = np.mean(losses)

    plt.figure()
    plt.plot(x, y, label='reconLoss')
    plt.plot(epoch_iters, epoch_losses, color='darkorange', label='epoch avg')
    plt.plot([np.min(iters), np.max(iters)], [mval, mval], color='darkorange', linestyle=':', label='window avg')

    plt.legend()
    plt.title('Short history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('{0}/history_short.png'.format(opt.save_dir), bbox_inches='tight')
    plt.close()
    
    ### Embedding figure
    plt.figure()
    colors = plt.get_cmap('plasma')(np.linspace(0, 1, embedding.shape[0]))
    plt.scatter(embedding[:,0], embedding[:,1], s = 2, color = colors)
    plt.xlim([-4, 4]) 
    plt.ylim([-4, 4])     
    plt.axis('equal')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('latent space embedding')
    plt.savefig('{0}/embedding.png'.format(opt.save_dir), bbox_inches='tight')
    plt.close()

    
def save_state(enc, dec, encD, decD, 
               optEnc, optDec, optEncD, optDecD, 
               logger, zAll, opt):
#         for saving and loading see:
#         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718
  
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
   