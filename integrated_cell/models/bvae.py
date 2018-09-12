import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

from integrated_cell import model_utils
from integrated_cell.utils import plots as plots
from integrated_cell import utils
from integrated_cell import SimpleLogger

from integrated_cell.models import base_model 

import importlib
import pickle
import os

# This is the trainer for the Beta-VAE

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



class Model(base_model.Model):
    def __init__(self, data_provider, 
                 n_epochs, 
                 n_channels,
                 n_latent_dim, 
                 n_classes, 
                 n_ref, 
                 gpu_ids, 
                 save_dir,
                 save_state_iter, 
                 save_progress_iter,
                 model_name,
                 kwargs_enc,
                 kwargs_dec, 
                 critRecon,
                 optimizer,
                 objective = 'B',
                 beta = 1,
                 c_max = 25,
                 gamma = 1000,
                 c_iters_max = 1.25E5,
                 provide_decoder_vars = True,
                 **kwargs):
        
        super(Model, self).__init__(data_provider, 
                                    n_epochs, 
                                    n_channels, 
                                    n_latent_dim, 
                                    n_classes, 
                                    n_ref, 
                                    gpu_ids, 
                                    save_dir = save_dir,
                                    save_state_iter = save_state_iter, 
                                    save_progress_iter = save_progress_iter,
                                    **kwargs)
        
        self.initialize(model_name, kwargs_enc, kwargs_dec, critRecon, optimizer)
        

        if objective == 'H':
            self.beta = beta
        elif objective == 'B':
            self.c_max = c_max
            self.gamma = gamma
            self.c_iters_max = c_iters_max

        self.provide_decoder_vars = provide_decoder_vars
        self.objective = objective

        
    def iteration(self):
        
        gpu_id = self.gpu_ids[0]
        
        enc, dec = self.enc, self.dec
        optEnc, optDec = self.optEnc, self.optDec
        critRecon, critZClass, critZRef = self.critRecon, self.critZClass, self.critZRef
        
        
        x, classes, ref = self.data_provider.get_sample()
        
        
        self.x.data.copy_(x)
        x = self.x
        
        classes = classes.type_as(x).long()
        
        if self.n_classes > 0:
            classes = classes.type_as(x).long()

        if self.n_ref > 0:
            ref = ref.type_as(x)

        #####################
        ### train autoencoder
        #####################

        enc.train(True)
        dec.train(True)
        
        optEnc.zero_grad()
        optDec.zero_grad()

        ### Forward passes
        zAll = enc(x)

        c = 0
        ### Update the class discriminator
        if self.n_classes > 0:
            classLoss = critZClass(zAll[c], classes)
            classLoss.backward(retain_graph=True)
            classLoss = classLoss.data[0]
            
            if self.provide_decoder_vars:
                zAll[c] = torch.log(utils.index_to_onehot(classes, self.data_provider.get_n_classes()) + 1E-8)
            
            c += 1
            
        ### Update the reference shape discriminator
        if self.n_ref > 0:
            refLoss = critZRef(zAll[c], ref)
            refLoss.backward(retain_graph=True)
            refLoss = refLoss.data[0]
            
            if self.provide_decoder_vars:
                zAll[c] = ref

            c += 1
                
        total_kld, dimension_wise_kld, mean_kld = kl_divergence(zAll[c][0], zAll[c][1])
        
        zLatent = zAll[c][0].data.cpu()
        
        zAll[c] = reparameterize(zAll[c][0], zAll[c][1])

        xHat = dec(zAll)

        ### Update the image reconstruction
        recon_loss = critRecon(xHat, x)

        if self.objective == 'H':
            beta_vae_loss = recon_loss + self.beta*total_kld
        elif self.objective == 'B':
            C = torch.clamp(torch.Tensor([self.c_max/self.c_iters_max*len(self.logger)]).type_as(x), 0, self.c_max)
            beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

        beta_vae_loss.backward()
        
        kld_loss = mean_kld.data[0]
        recon_loss = recon_loss.data[0]

        optEnc.step()
        optDec.step()

        errors = (recon_loss,)
        if self.n_classes > 0:
            errors += (classLoss,)

        if self.n_ref > 0:
            errors += (refLoss,)

        errors += (kld_loss,)
        

        errors = [error.cpu() for error in errors]
        return errors, zLatent


    def initialize(self, model_name, kwargs_enc, kwargs_dec, critRecon, optimizer):

        gpu_id = self.gpu_ids[0]
        
        model_provider = importlib.import_module("integrated_cell.networks." + model_name)

        enc = model_provider.Enc(self.n_latent_dim, self.n_classes, self.n_ref, self.n_channels, self.gpu_ids, **kwargs_enc)
        dec = model_provider.Dec(self.n_latent_dim, self.n_classes, self.n_ref, self.n_channels, self.gpu_ids, **kwargs_dec)
       
        enc.apply(model_utils.weights_init)
        dec.apply(model_utils.weights_init)
       
        enc.cuda(gpu_id)
        dec.cuda(gpu_id)
       
        if optimizer == 'RMSprop':
            optEnc = optim.RMSprop(enc.parameters(), lr=self.lrEnc)
            optDec = optim.RMSprop(dec.parameters(), lr=self.lrDec)
        elif optimizer == 'adam':

            optEnc = optim.Adam(enc.parameters(), lr=self.lrEnc, **self.kwargs_optim)
            optDec = optim.Adam(dec.parameters(), lr=self.lrDec, **self.kwargs_optim)

        self.enc = enc
        self.dec = dec
        self.optEnc = optEnc
        self.optDec = optDec
            
        columns = ('epoch', 'iter', 'reconLoss',)
        print_str = '[%d][%d] reconLoss: %.6f'

        if self.n_classes > 0:
            columns += ('classLoss',)
            print_str += ' classLoss: %.6f'

        if self.n_ref > 0:
            columns += ('refLoss',)
            print_str += ' refLoss: %.6f'

        columns += ('kldLoss', 'time')
        print_str += ' kld: %.6f time: %.2f'

        self.logger = SimpleLogger(columns,  print_str)

        self.critRecon = eval('nn.' + critRecon + '(size_average=False)')
        self.critZClass = nn.NLLLoss(size_average=False)
        self.critZRef = nn.MSELoss(size_average=False)
        
        
    def load(self, save_dir):
        gpu_id = self.gpu_ids[0]
        
        if os.path.exists('{0}/enc.pth'.format(save_dir)):
            print('Loading from ' + save_dir)

            self.logger = pickle.load(open( '{0}/logger.pkl'.format(save_dir), "rb" ))
            
            model_utils.load_state(self.enc, self.optEnc, '{0}/enc.pth'.format(save_dir), gpu_id)
            model_utils.load_state(self.dec, self.optDec, '{0}/dec.pth'.format(save_dir), gpu_id)

    
    def save(self, save_dir):
    #         for saving and loading see:
    #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        gpu_id = self.gpu_ids[0]

        model_utils.save_state(self.enc, self.optEnc, '{0}/enc.pth'.format(save_dir), gpu_id)
        model_utils.save_state(self.dec, self.optDec, '{0}/dec.pth'.format(save_dir), gpu_id)

        pickle.dump(self.logger, open('{0}/logger.pkl'.format(save_dir), 'wb'))

