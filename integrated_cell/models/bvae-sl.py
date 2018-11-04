import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


from integrated_cell import model_utils
from integrated_cell.utils import plots as plots
import integrated_cell.utils as utils

# This is the trainer for the Beta-VAE

from integrated_cell import utils
from integrated_cell.models import bvae
from integrated_cell.models import base_model 
from integrated_cell import SimpleLogger

import os
import importlib
import pickle

from .. import losses





class Model(bvae.Model):
    def __init__(self, 
                 enc,
                 dec,
                 opt_enc,
                 opt_dec,
                 crit_loss,
                 n_epochs, 
                 gpu_ids,
                 save_dir,
                 logger,
                 data_provider
                 save_progres_iter = 1,
                 save_state_iter = 10
                 beta = 1, 
                 c_max = 25, 
                 c_iters_max = 1.2E5, 
                 gamma = 1000, 
                 objective = 'H', 
                 lambda_loss = 1E-4,
                 lambda_ref_loss = 1,
                 lambda_class_loss = 1,
                 provide_decoder_vars = False
    ):
        
        super(Model, self).__init__(
                            data_provider, 
                            n_epochs, 
                            n_channels, 
                            n_latent_dim, 
                            n_classes, 
                            n_ref, 
                            gpu_ids, 
                            save_dir = save_dir,
                            save_state_iter = save_state_iter, 
                            save_progress_iter = save_progress_iter
        )
        
        self.enc = enc
        self.dec = dec
        self.opt_enc = opt_enc
        self.opt_dec = opt_dec
        self.logger = logger
        self.crit_loss = crit_loss
        self.gpu_ids = gpu_ids
        
        self.objective = objective
        if objective == 'H':
            self.beta = beta
        elif objective == 'B':
            self.c_max = c_max
            self.gamma = gamma
            self.c_iters_max = c_iters_max

        self.provide_decoder_vars = provide_decoder_vars
        
        self.lambda_loss = lambda_loss
        self.lambda_ref_loss = lambda_ref_loss
        self.lambda_class_loss = lambda_class_loss
        
    def iteration(self):
        gpu_id = self.gpu_ids[0]

        enc, dec = self.enc, self.dec
        optEnc, optDec = self.optEnc, self.optDec
        critRecon, critZClass, critZRef = self.critRecon, self.critZClass, self.critZRef
        
        #do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)

        ###update the discriminator
        #maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))
        x, classes, ref = self.data_provider.get_sample()
        
        self.x.data.copy_(x)
        x = self.x

        y_xFake = self.y_xFake
        y_zReal = self.y_zReal


        
        if self.n_classes == 0:
            y_xReal = self.y_xReal
            y_zFake = self.y_zFake
        else:
            classes = classes.type_as(x).long()

            y_xReal = classes
            y_zFake = classes

        if self.n_ref > 0:
            ref = ref.type_as(x)

        for p in enc.parameters():
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

            
            
        optEnc.zero_grad()
        optDec.zero_grad()

        #####################
        ### train autoencoder
        #####################

        ### Forward passes
        zAll, activations = enc(x)

        c = 0
        ### Update the class discriminator
        if self.n_classes > 0:
            classLoss = critZClass(zAll[c], classes)*self.lambda_class_loss
            classLoss.backward(retain_graph=True)
            classLoss = classLoss.data[0]
            
            if self.provide_decoder_vars:
                zAll[c] = torch.log(utils.index_to_onehot(classes, self.n_classes) + 1E-8)
            
            c += 1
            
        ### Update the reference shape discriminator
        if self.n_ref > 0:
            refLoss = critZRef(zAll[c], ref)*self.lambda_ref_loss
            refLoss.backward(retain_graph=True)
            refLoss = refLoss.data[0]
            
            if self.provide_decoder_vars:
                zAll[c] = ref

            c += 1

        total_kld, dimension_wise_kld, mean_kld = bvae.kl_divergence(zAll[c][0], zAll[c][1])
        
        zLatent = zAll[c][0].data.cpu()

        zAll[c] = bvae.reparameterize(zAll[c][0], zAll[c][1])

        xHat = dec(zAll)

        ### Update the image reconstruction
        recon_loss = critRecon(xHat, x)
        
        if self.objective == 'H':
            beta_vae_loss = recon_loss + self.beta*total_kld
        elif self.objective == 'B':
            C = torch.clamp(torch.Tensor([self.c_max/self.c_iters_max*len(self.logger)]).type_as(x), 0, self.c_max)
            beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()

        beta_vae_loss.backward(retain_graph=True)
        
        for p in enc.parameters(): 
            p.requires_grad = False
            
        _, activations_hat = enc(xHat)

        self_loss = torch.tensor(0).type_as(x)
        for activation_hat, activation in zip(activations_hat, activations):          
            self_loss += critRecon(activation_hat, activation.detach())

        (self_loss*self.lambda_loss).backward()
    
        for p in enc.parameters(): 
            p.requires_grad = False
    
        optEnc.step()
        optDec.step()
        
        kld_loss = total_kld.item()
        recon_loss = recon_loss.item()
        self_loss = self_loss.item()
        
        errors = [recon_loss,]
        if self.n_classes > 0:
            errors += [classLoss,]

        if self.n_ref > 0:
            errors += [refLoss,]

        errors += [kld_loss, self_loss]
        
        return errors, zLatent

    def save(self):
        embeddings = torch.cat(self.zAll,0).cpu().numpy()
        
        save_f(self.enc, self.dec, self.opt_enc, self.opt_dec, self.logger, embeddings, self.gpu_id)
    
def load(net_name, kwargs_enc, kwargs_dec, optim_name, optim_enc_kwargs, optim_dec_kwargs, save_dir, gpu_ids):
    
    model_provider = importlib.import_module("integrated_cell.networks." + net_name)
    enc = model_provider.Enc(gpu_ids = gpu_ids, **kwargs_enc)
    dec = model_provider.Dec(gpu_ids = gpu_ids, **kwargs_dec)
    
    enc.apply(model_utils.weights_init)
    dec.apply(model_utils.weights_init)
    
    enc.cuda(gpu_ids[0])
    dec.cuda(gpu_ids[0])
    
    optimizer_constructor = eval("torch.optim." + optimizer)
    
    opt_enc = optimizer_constructor(enc.parameters(), **optim_enc_kwargs)
    opt_dec = optimizer_constructor(dec.parameters(), **optim_dec_kwargs)
    
    columns = ('epoch', 'iter', 'reconLoss',)
    print_str = '[%d][%d] reconLoss: %.6f'

    if self.n_classes > 0:
        columns += ('classLoss',)
        print_str += ' classLoss: %.6f'

    if self.n_ref > 0:
        columns += ('refLoss',)
        print_str += ' refLoss: %.6f'

    columns += ('kldLoss', 'selfLoss' 'time')
    print_str += ' kld: %.6f sLoss: %.6f time: %.2f'

    logger = SimpleLogger(columns,  print_str)
    
    enc_path = "{}/enc.pth".format(save_dir)
    dec_path = "{}/dec.pth".format(save_dir)
    logger_path = "{}/enc.logger".format(save_dir)
        
    if os.path.exists(enc_path):
        print('Loading from ' + save_dir)

        logger = pickle.load(open(logger_path, "rb"))
        
        model_utils.load_state(enc, opt_enc, '{0}/enc_{1}.pth'.format(save_dir, int(len(logger))), gpu_id)
        model_utils.load_state(dec, opt_dec, '{0}/dec_{1}.pth'.format(save_dir, int(len(logger))), gpu_id)
        
    return enc, dec, opt_enc, opt_dec, logger

def save_f(enc, dec, opt_enc, opt_dec, logger, embeddings, gpu_id):
    #for saving and loading see:
    #https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

    model_utils.save_state(enc, opt_enc, '{0}/enc_{1}.pth'.format(save_dir, int(len(logger))), gpu_id)
    model_utils.save_state(dec, opt_dec, '{0}/dec_{1}.pth'.format(save_dir, int(len(logger))), gpu_id)

    pickle.dump(embedding, open('{0}/embedding_{1}.pth'.format(save_dir, int(len(logger))), 'wb'))

    pickle.dump(self.logger, open('{0}/logger.pkl'.format(save_dir), 'wb'))


