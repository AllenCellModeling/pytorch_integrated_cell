import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


from integrated_cell.model_utils import *
import integrated_cell.utils as utils
from integrated_cell.utils import plots as plots

# This is the base class for trainers

def reparameterize(mu, log_var, add_noise = True):
    if add_noise:
        std = log_var.div(2).exp()
        eps = torch.randn_like(std)
        out = eps.mul(std).add_(mu)
    else:
        out = mu
        
    return out

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
        
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def get_channel_subsets(x_batch, y_batch, max_channels):
    
    x_in, x_out, y_in, y_out = list(), list(), list(), list()
    
    n_channels = np.random.randint(1, x_batch.shape[1], [x_batch.shape[0]])
    
    rm_inds = np.where(np.cumsum(n_channels) > max_channels)[0]
    if len(rm_inds) > 0:
        n_channels[rm_inds[0]:] = 0
    
    for x, y, n_channel in zip(x_batch, y_batch, n_channels):
        
        if n_channel == 0:
            break
        
        in_inds = np.random.choice(np.arange(0, x_batch.shape[1]), n_channel)
        
        remaining_inds = np.ones(x_batch.shape[1])
        remaining_inds[in_inds] = 0
        remaining_inds = np.where(remaining_inds)[0]
        
        out_inds = np.random.choice(remaining_inds, 1)
        
        x_in += [x[in_inds]]
        x_out += [x[out_inds]]
        
        y_in += [y[in_inds]]      
        y_out += [y[out_inds]]   
        
    return x_in, x_out, y_in, y_out



class Model(object):
    def __init__(self, data_provider, n_channels, batch_size, n_latent_dim, n_classes, n_ref, gpu_ids, max_channels = 48):

        self.batch_size = batch_size
        self.n_channels = n_channels

        self.n_ref = n_ref
        self.n_latent_dim = n_latent_dim
        
        self.gpu_ids = gpu_ids
        
        self.n_classes = n_classes
        self.n_classes = data_provider.get_images([0], 'train').shape[1] -1 + data_provider.get_n_classes()

        #because we sample the number of channels we need, we set an upper bound so we dont run out of memory
        self.max_channels = max_channels
        
        gpu_id = self.gpu_ids[0]
        
        
    def get_data(self, data_provider, inds, train_or_test = 'train'):
        
        gpu_id = self.gpu_ids[0]
        
        x = data_provider.get_images(inds,train_or_test).cuda(gpu_id)

        classes_tmp = data_provider.get_classes(inds,'train').type_as(x).long() + x.shape[1]-1

        n_classes = self.n_classes

        classes = torch.zeros(x.shape[0], x.shape[1], n_classes).type_as(x)
        for i in range(classes.shape[0]):
            classes[i] = utils.index_to_onehot( torch.cat([torch.arange(x.shape[1]-1), torch.Tensor([classes_tmp[i]])]) , n_classes)

        x_in, x_out, y_in, y_out = get_channel_subsets(x, classes, self.max_channels)


        y_in = torch.cat(y_in, 0)
        y_out = torch.cat(y_out, 0)

        x_out = torch.cat([x.unsqueeze(0) for x in x_out],0)

        return x_in, x_out, y_in, y_out

    def iteration(self,
                  enc, dec,
                  optEnc, optDec,
                  critRecon, critZClass, critZRef,
                  data_provider, opt):

        rand_inds_encD = np.random.permutation(opt.ndat)
        inds = rand_inds_encD[0:self.batch_size]

        #the variable "class" channel should be the last channel
        x_in, x_out, y_in, y_out = self.get_data(data_provider, inds, train_or_test = 'train')
        
        
    
        #####################
        ### train autoencoder
        #####################

        enc.train(True)
        dec.train(True)
        
        optEnc.zero_grad()
        optDec.zero_grad()
        
        z = enc(x_in, y_in)
        
        x_out_hat = dec(z, y_out)
        
        loss = critRecon(x_out_hat, x_out)
        
        loss.backward()
        
        optEnc.step()
        optDec.step()
        
        return [loss.cpu().item()], z.view(x_out.shape[0], 1024*4*3*2).cpu().detach()


    def load(self, model_name, opt):

        model_provider = importlib.import_module("integrated_cell.networks." + model_name)

        enc = model_provider.Enc(n_classes = self.n_classes, gpu_ids = self.gpu_ids, n_proj_dims = self.n_latent_dim, **opt.kwargs_enc)
        dec = model_provider.Dec(n_classes = self.n_classes, gpu_ids = self.gpu_ids, **opt.kwargs_dec)
       
        enc.apply(weights_init)
        dec.apply(weights_init)
       
        gpu_id = self.gpu_ids[0]

        enc.cuda(gpu_id)
        dec.cuda(gpu_id)
       
        if opt.optimizer == 'RMSprop':
            optEnc = optim.RMSprop(enc.parameters(), lr=opt.lrEnc)
            optDec = optim.RMSprop(dec.parameters(), lr=opt.lrDec)
        elif opt.optimizer == 'adam':

            optEnc = optim.Adam(enc.parameters(), lr=opt.lrEnc, **opt.kwargs_optim)
            optDec = optim.Adam(dec.parameters(), lr=opt.lrDec, **opt.kwargs_optim)

        columns = ('epoch', 'iter', 'reconLoss',)
        print_str = '[%d][%d] reconLoss: %.6f'

#         if self.n_classes > 0:
#             columns += ('classLoss',)
#             print_str += ' classLoss: %.6f'

#         if self.n_ref > 0:
#             columns += ('refLoss',)
#             print_str += ' refLoss: %.6f'

        columns += ('time',)
        print_str += ' time: %.2f'

        logger = SimpleLogger(columns,  print_str)

        if os.path.exists('{0}/enc.pth'.format(opt.save_dir)):
            print('Loading from ' + opt.save_dir)

            load_state(enc, optEnc, '{0}/enc.pth'.format(opt.save_dir), gpu_id)
            load_state(dec, optDec, '{0}/dec.pth'.format(opt.save_dir), gpu_id)

            logger = pickle.load(open( '{0}/logger.pkl'.format(opt.save_dir), "rb" ))

            this_epoch = max(logger.log['epoch']) + 1
            iteration = max(logger.log['iter'])

        models = dict()
        models['enc'] = enc
        models['dec'] = dec

        optimizers = dict()
        optimizers['optEnc'] = optEnc
        optimizers['optDec'] = optDec

        criterions = dict()
        criterions['critRecon'] = eval('nn.' + opt.critRecon + '()')
        criterions['critZClass'] = nn.NLLLoss(size_average=False)
        criterions['critZRef'] = nn.MSELoss(size_average=False)

        if opt.latentDistribution == 'uniform':
            from integrated_cell.model_utils import sampleUniform as latentSample
        elif opt.latentDistribution == 'gaussian':
            from integrated_cell.model_utils import sampleGaussian as latentSample

        self.latentSample = latentSample
        
        self.models = models
        self.optimizers = optimizers
        self.criterions = criterions
        self.logger = logger
        self.opt = opt
        
        return models, optimizers, criterions, logger

    def save(self, enc, dec,
                   optEnc, optDec,
                   logger, zAll, opt):
    #         for saving and loading see:
    #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        gpu_id = self.gpu_ids[0]

        save_state(enc, optEnc, '{0}/enc.pth'.format(opt.save_dir), gpu_id)
        save_state(dec, optDec, '{0}/dec.pth'.format(opt.save_dir), gpu_id)

        pickle.dump(zAll, open('{0}/embedding.pkl'.format(opt.save_dir), 'wb'))
        pickle.dump(logger, open('{0}/logger.pkl'.format(opt.save_dir), 'wb'))




    def save_progress(self, enc, dec, data_provider, logger, embedding, opt):
        gpu_id = self.gpu_ids[0]

        epoch = max(logger.log['epoch'])

        enc.train(False)
        dec.train(False)

    #     pdb.set_trace()
        train_classes = data_provider.get_classes(np.arange(0, data_provider.get_n_dat('train')), 'train')
        _, train_inds = np.unique(train_classes.numpy(), return_index=True)

        x_in, x_out, y_in, y_out = self.get_data(data_provider, train_inds, train_or_test = 'train')
        
        with torch.no_grad():
            xHat = dec(enc(x_in, y_in), y_out)
            
        imgX = tensor2img(x_out.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgTrainOut = np.concatenate((imgX, imgXHat), 0)

        test_classes = data_provider.get_classes(np.arange(0, data_provider.get_n_dat('test')), 'test')
        _, test_inds = np.unique(test_classes.numpy(), return_index=True)

        x_in, x_out, y_in, y_out = self.get_data(data_provider, test_inds, train_or_test = 'test')
        
        with torch.no_grad():
            xHat = dec(enc(x_in, y_in), y_out)

        z = list()
        if self.n_classes > 0:
            class_var = torch.Tensor(data_provider.get_classes(test_inds, 'test', 'one_hot').float()).cuda(gpu_id)
            class_var = (class_var-1) * 25
            z.append(class_var)

        if self.n_ref > 0:
            ref_var = torch.Tensor(data_provider.get_n_classes(), self.n_ref).normal_(0,1).cuda(gpu_id)
            z.append(ref_var)

        loc_var = torch.Tensor(data_provider.get_n_classes(), self.n_latent_dim).normal_(0,1).cuda(gpu_id)
        z.append(loc_var)

        # x_z = dec(z, y_out)

        imgX = tensor2img(x_out.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        # imgX_z = tensor2img(x_z.data.cpu())
        # imgTestOut = np.concatenate((imgX, imgXHat, imgX_z), 0)

        imgTestOut = np.concatenate((imgX, imgXHat), 0)
        
        imgOut = np.concatenate((imgTrainOut, imgTestOut))

        scipy.misc.imsave('{0}/progress_{1}.png'.format(opt.save_dir, int(epoch)), imgOut)

        enc.train(True)
        dec.train(True)

        # pdb.set_trace()
        # zAll = torch.cat(zAll,0).cpu().numpy()

        pickle.dump(embedding, open('{0}/embedding_tmp.pkl'.format(opt.save_dir), 'wb'))
        pickle.dump(logger, open('{0}/logger_tmp.pkl'.format(opt.save_dir), 'wb'))

        ### History
        plots.history(logger, '{0}/history.png'.format(opt.save_dir))
        
        ### Short History
        plots.short_history(logger, '{0}/history_short.png'.format(opt.save_dir))
        
        ### Embedding figure
        plots.embeddings(embedding, '{0}/embedding.png'.format(opt.save_dir))

        xHat = None
        x = None

