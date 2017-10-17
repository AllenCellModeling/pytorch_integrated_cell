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
import importlib

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
        
def load_embeddings(embeddings_path, enc=None, dp=None, opt=None):

    if os.path.exists(embeddings_path):
        embeddings = torch.load(embeddings_path)
    else:
        embeddings = get_latent_embeddings(enc, dp, opt)
        torch.save(embeddings, embeddings_path)
    
    return embeddings        
        
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
            print(str(i) + '/' + str(len(data_iter)))
            x = Variable(dp.get_images(data_iter[i], mode).cuda(gpu_id), volatile=True)
            zAll = enc(x)
            
            embeddings.index_copy_(0, torch.LongTensor(data_iter[i]), zAll[-1].data[:].cpu())
        
        embedding[mode] = embeddings
        
    return embedding

def load_data_provider(data_path, im_dir, dp_module):
    DP = importlib.import_module("data_providers." + dp_module)

    if os.path.exists(data_path):
        dp = torch.load(data_path)
    else:
        dp = DP.DataProvider(im_dir)
        torch.save(dp, data_path)
        
    return dp
        
def load_model(model_name, opt):
    model_provider = importlib.import_module("models." + model_name)
 
    enc = model_provider.Enc(opt.nlatentdim, opt.nClasses, opt.nRef, opt.nch, opt.gpu_ids, opt)
    dec = model_provider.Dec(opt.nlatentdim, opt.nClasses, opt.nRef, opt.nch, opt.gpu_ids, opt)
    encD = model_provider.EncD(opt.nlatentdim, opt.nClasses+1, opt.gpu_ids, opt)
    decD = model_provider.DecD(opt.nClasses+1, opt.nch, opt.gpu_ids, opt)
    
    enc.apply(weights_init)
    dec.apply(weights_init)
    encD.apply(weights_init)
    decD.apply(weights_init)

    gpu_id = opt.gpu_ids[0]

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
    print_str += ' mmEncD: %.6f encD: %.6f  mmDecD: %.6f decD: %.6f time: %.2f'

    
    logger = SimpleLogger.SimpleLogger(columns,  print_str)

    if os.path.exists('./{0}/enc.pth'.format(opt.save_dir)):
        print('Loading from ' + opt.save_dir)
                    
        load_state(enc, optEnc, './{0}/enc.pth'.format(opt.save_dir), gpu_id)
        load_state(dec, optDec, './{0}/dec.pth'.format(opt.save_dir), gpu_id)
        load_state(encD, optEncD, './{0}/encD.pth'.format(opt.save_dir), gpu_id)
        load_state(decD, optDecD, './{0}/decD.pth'.format(opt.save_dir), gpu_id)
                            
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
    criterions['critRecon'] = eval('nn.' + opt.critRecon + '()')
    criterions['critZClass'] = nn.NLLLoss()    
    criterions['critZRef'] = nn.MSELoss()
    
    if opt.nClasses > 0:
        criterions['critDecD'] = nn.CrossEntropyLoss()
        criterions['critEncD'] = nn.CrossEntropyLoss()
    else:
        criterions['critEncD'] = nn.BCEWithLogitsLoss()
        criterions['critDecD'] = nn.BCEWithLogitsLoss()
        
    if opt.latentDistribution == 'uniform':
        from model_utils import sampleUniform as latentSample
        
    elif opt.latentDistribution == 'gaussian':
        from model_utils import sampleGaussian as latentSample
        

    opt.latentSample = latentSample   
    
    return models, optimizers, criterions, logger, opt

def load_state(model, optimizer, path, gpu_id):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model'])
    model.cuda(gpu_id)

    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)
            
def save_state(model, optimizer, path, gpu_id):
    
    model.cpu()
    optimizer.state = set_gpu_recursive(optimizer.state, -1)
    
    checkpoint = {'model': model.state_dict(),
              'optimizer': optimizer.state_dict()}
    
    torch.save(checkpoint, path)
    
    model.cuda(gpu_id)
    optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)
    
def save_state_all(enc, dec, encD, decD, 
               optEnc, optDec, optEncD, optDecD, 
               logger, zAll, opt):
#         for saving and loading see:
#         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718
  
    gpu_id = opt.gpu_ids[0]
    
    
    save_state(enc, optEnc, './{0}/enc.pth'.format(opt.save_dir), gpu_id)
    save_state(dec, optDec, './{0}/dec.pth'.format(opt.save_dir), gpu_id)
    save_state(encD, optEncD, './{0}/encD.pth'.format(opt.save_dir), gpu_id)
    save_state(decD, optDecD, './{0}/decD.pth'.format(opt.save_dir), gpu_id)
    
    pickle.dump(zAll, open('./{0}/embedding.pkl'.format(opt.save_dir), 'wb'))
    pickle.dump(logger, open('./{0}/logger.pkl'.format(opt.save_dir), 'wb'))
   
    
def save_progress(enc, dec, dataProvider, logger, embedding, opt):
    
    gpu_id = opt.gpu_ids[0]
    
    epoch = max(logger.log['epoch'])
    
    enc.train(False)
    dec.train(False)

    x = Variable(dataProvider.get_images(np.arange(0,10),'train').cuda(gpu_id), volatile=True)
    
    # try:
    xHat = dec(enc(x))
    # except:
    #     xHat = dec(enc(x)[0])
        
    imgX = tensor2img(x.data.cpu())
    imgXHat = tensor2img(xHat.data.cpu())
    imgTrainOut = np.concatenate((imgX, imgXHat), 0)

    x = Variable(dataProvider.get_images(np.arange(0,10),'test').cuda(gpu_id), volatile=True)
    # try:
    xHat = dec(enc(x))
    # except:
    #     xHat = dec(enc(x)[0])
    
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
    history = int(len(logger.log['epoch'])/2)
    
    if history > 10000:
        history = 10000
    
    ydat = [logger.log['encDLoss'], logger.log['minimaxEncDLoss'], logger.log['decDLoss'], logger.log['minimaxDecDLoss']]
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
    
    xHat = None
    x = None

def maybe_save(epoch, epoch_next, models, optimizers, logger, zAll, dp, opt):
    saved = False
    if epoch != epoch_next and ((epoch_next % opt.saveProgressIter) == 0 or (epoch_next % opt.saveStateIter) == 0):

        zAll = torch.cat(zAll,0).cpu().numpy()

        if (epoch_next % opt.saveProgressIter) == 0:
            print('saving progress')
            save_progress(models['enc'], models['dec'], dp, logger, zAll, opt)

        if (epoch_next % opt.saveStateIter) == 0:
            print('saving state')
            save_state_all(**models, **optimizers, logger=logger, zAll=zAll, opt=opt)

        saved = True
        
    return saved    
