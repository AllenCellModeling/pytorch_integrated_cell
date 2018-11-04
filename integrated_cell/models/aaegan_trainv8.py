import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

import importlib

import integrated_cell.model_utils as model_utils
import integrated_cell.utils as utils

from integrated_cell.models import base_model 

from integrated_cell.SimpleLogger import SimpleLogger

import os

from .. import losses

import pickle

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
                 kwargs_encD,
                 kwargs_decD,
                 critRecon,
                 critAdv,
                 optimizer,
                 size_average_losses = False,
                 **kwargs
                ):
        
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
 

        
        self.size_average_losses = size_average_losses
        self.initialize(model_name, kwargs_enc, kwargs_dec, kwargs_encD, kwargs_decD, critRecon, critAdv, optimizer)

        
        
    def iteration(self):
        
        gpu_id = self.gpu_ids[0]

        enc, dec, encD, decD = self.enc, self.dec, self.encD, self.decD
        optEnc, optDec, optEncD, optDecD = self.optEnc, self.optDec, self.optEncD, self.optDecD
        critRecon, critZClass, critZRef, critDecD, critEncD = self.critRecon, self.critZClass, self.critZRef, self.critDecD, self.critEncD
        
        
        #do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)
        encD.train(True)
        decD.train(True)

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
            self.classes.data.copy_(classes)
            classes = self.classes

            y_xReal = classes
            y_zFake = classes

        if self.n_ref > 0:
            self.ref.data.copy_(ref)
            ref = self.ref

        for p in encD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for p in decD.parameters():
            p.requires_grad = True

        for p in enc.parameters():
            p.requires_grad = False

        for p in dec.parameters():
            p.requires_grad = False

        zAll = enc(x)

        if self.provide_decoder_vars:
            c = 0
            if self.n_classes > 0:
                zAll[c] = torch.log(utils.index_to_onehot(classes, self.data_provider.get_n_classes()) + 1E-8)
                c += 1   
            
            if self.n_ref > 0:
                zAll[c] = ref
            c += 1
            
        for var in zAll:
            var.detach_()

        xHat = dec(zAll)

        self.zReal.data.normal_()
        zReal = self.zReal
        zFake = zAll[-1]

        optEnc.zero_grad()
        optDec.zero_grad()
        optEncD.zero_grad()
        optDecD.zero_grad()

        ###############
        ### train encD
        ###############

        ### train with real
        yHat_zReal = encD(zReal)
        errEncD_real = critEncD(yHat_zReal, y_zReal)

        ### train with fake
        yHat_zFake = encD(zFake)
        errEncD_fake = critEncD(yHat_zFake, y_zFake)

        encDLoss = (errEncD_real + errEncD_fake)/2
        encDLoss.backward(retain_graph=True)

        optEncD.step()

        encDLoss = encDLoss.data[0]

        ##############
        ### Train discriminators
        ##############

        yHat_xReal = decD(x)

        ### train with real
        errDecD_real = critDecD(yHat_xReal, y_xReal)

        ### train with fake, reconstructed
        yHat_xFake = decD(xHat)
        errDecD_fake = critDecD(yHat_xFake, y_xFake)

        ### train with fake, sampled and decoded
        zAll[-1] = zReal

        yHat_xFake2 = decD(dec(zAll))
        errDecD_fake2 = critDecD(yHat_xFake2, y_xFake)

        decDLoss = (errDecD_real + (errDecD_fake + errDecD_fake2)/2)/2
        decDLoss.backward(retain_graph=True)
        optDecD.step()

        decDLoss = decDLoss.data[0]

        errEncD_real = None
        errEncD_fake = None

        errDecD_real = None
        errDecD_fake = None
        errDecD_fake2 = None

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

        #####################
        ### train autoencoder
        #####################

        ### Forward passes
        zAll = enc(x)

        c = 0
        ### Update the class discriminator
        if self.n_classes > 0:
            classLoss = critZClass(zAll[c], classes)
            classLoss.mul(self.lambda_class_loss).backward(retain_graph=True)
            classLoss = classLoss.data[0]
            
            if self.provide_decoder_vars:
                zAll[c] = torch.log(utils.index_to_onehot(classes, self.data_provider.get_n_classes()) + 1E-8)
            
            c += 1

        ### Update the reference shape discriminator
        if self.n_ref > 0:
            refLoss = critZRef(zAll[c], ref)
            refLoss.mul(self.lambda_ref_loss).backward(retain_graph=True)
            refLoss = refLoss.data[0]
            
            if self.provide_decoder_vars:
                zAll[c] = ref
            
            c += 1

        xHat = dec(zAll)

        ### Update the image reconstruction
        reconLoss = critRecon(xHat, x)
        reconLoss.backward(retain_graph=True)
        reconLoss = reconLoss.data[0]

        ### update wrt encD
        yHat_zFake = encD(zAll[c])
        minimaxEncDLoss = critEncD(yHat_zFake, y_zReal)
        (minimaxEncDLoss.mul(self.lambda_encD_loss)).backward(retain_graph=True)
        minimaxEncDLoss = minimaxEncDLoss.data[0]

        optEnc.step()

        for p in enc.parameters():
            p.requires_grad = False

        ### update wrt decD(dec(enc(X)))
        yHat_xFake = decD(xHat)
        minimaxDecDLoss = critDecD(yHat_xFake, y_xReal)
        (minimaxDecDLoss.mul(self.lambda_decD_loss).div(2)).backward(retain_graph=True)
        minimaxDecDLoss = minimaxDecDLoss.data[0]
        yHat_xFake = None

        ### update wrt decD(dec(Z))

        c = 0
        #if we have classes, create random classes, generate images of random classes
        if self.n_classes > 0:
            shuffle_inds = np.arange(0, zAll[0].size(0))

            classes_one_hot = ((utils.index_to_onehot(classes, self.n_classes) - 1) * 25).type_as(zAll[c].data).type_as(x)

            np.random.shuffle(shuffle_inds)
            zAll[c] = classes_one_hot[shuffle_inds,:]
            y_xReal = y_xReal[torch.LongTensor(shuffle_inds)]

            c +=1

        if self.n_ref > 0:
            zAll[c].data.normal_()


        #sample random positions in the localization space
        self.zReal.data.normal_()
        zAll[-1] = self.zReal

        xHat = dec(zAll)

        yHat_xFake2 = decD(xHat)
        minimaxDecDLoss2 = critDecD(yHat_xFake2, y_xReal)
        (minimaxDecDLoss2.mul(self.lambda_decD_loss).div(2)).backward(retain_graph=True)
        minimaxDecDLoss2 = minimaxDecDLoss2.data[0]
        yHat_xFake2 = None

        minimaxDecLoss = (minimaxDecDLoss+minimaxDecDLoss2)/2

        optDec.step()

        errors = (reconLoss,)
        if self.n_classes > 0:
            errors += (classLoss,)

        if self.n_ref > 0:
            errors += (refLoss,)

        errors += (minimaxEncDLoss, encDLoss, minimaxDecLoss, decDLoss)
        errors = [error.cpu() for error in errors]

        return errors, zFake.data.cpu()

    def initialize(self, model_name, kwargs_enc, kwargs_dec, kwargs_encD, kwargs_decD, critRecon, critAdv, optimizer):
        gpu_id = self.gpu_ids[0]
        model_provider = importlib.import_module("integrated_cell.networks." + model_name)
        
        
        
        kwargs_enc_tmp = {'nLatentDim': self.n_latent_dim,
                         'nClasses': self.n_classes,
                         'nRef': self.n_ref,
                         'nch': self.n_channels,
                         'gpu_ids': self.gpu_ids}
        
        for k in kwargs_enc: kwargs_enc_tmp[k] = kwargs_enc[k]
        
        self.enc = model_provider.Enc(**kwargs_enc_tmp)
        
        kwargs_dec_tmp = {'nLatentDim': self.n_latent_dim,
                 'nClasses': self.n_classes,
                 'nRef': self.n_ref,
                 'nch': self.n_channels,
                 'gpu_ids': self.gpu_ids}
        
        for k in kwargs_dec: kwargs_dec_tmp[k] = kwargs_dec[k]
        
        self.dec = model_provider.Dec(**kwargs_dec_tmp)
        

        self.encD = model_provider.EncD(self.n_latent_dim, self.n_classes+1, self.gpu_ids, **kwargs_encD)

        self.decD = model_provider.DecD(self.n_classes+1, self.n_channels, self.gpu_ids, **kwargs_decD)

        self.enc.apply(model_utils.weights_init)
        self.dec.apply(model_utils.weights_init)
        self.encD.apply(model_utils.weights_init)
        self.decD.apply(model_utils.weights_init)

        self.enc.cuda(gpu_id)
        self.dec.cuda(gpu_id)
        self.encD.cuda(gpu_id)
        self.decD.cuda(gpu_id)

        if optimizer == 'RMSprop':
            self.optEnc = optim.RMSprop(self.enc.parameters(), lr=self.lrEnc)
            self.optDec = optim.RMSprop(self.dec.parameters(), lr=self.lrDec)
            self.optEncD = optim.RMSprop(self.encD.parameters(), lr=self.lrEncD)
            self.optDecD = optim.RMSprop(self.decD.parameters(), lr=self.lrDecD)
        elif optimizer == 'Adam':
            self.optEnc = optim.Adam(self.enc.parameters(), lr=self.lrEnc, **self.kwargs_optim)
            self.optDec = optim.Adam(self.dec.parameters(), lr=self.lrDec, **self.kwargs_optim)
            self.optEncD = optim.Adam(self.encD.parameters(), lr=self.lrEncD, **self.kwargs_optim)
            self.optDecD = optim.Adam(self.decD.parameters(), lr=self.lrDecD, **self.kwargs_optim)
   
        self.critRecon = eval('nn.' + critRecon + '(size_average=' + str(bool(self.size_average_losses)) + ')')
        self.critZClass = nn.NLLLoss(size_average=self.size_average_losses)
        self.critZRef = nn.MSELoss(size_average=self.size_average_losses)
        
        if self.n_classes > 0:
            self.critDecD = nn.CrossEntropyLoss(size_average=self.size_average_losses)
            self.critEncD = nn.CrossEntropyLoss(size_average=self.size_average_losses)
        else:
            self.critDecD = eval(critAdv + '(size_average=' + str(bool(self.size_average_losses)) + ')')
            self.critEncD = eval(critAdv + '(size_average=' + str(bool(self.size_average_losses)) + ')')

        if self.latent_distribution == 'uniform':
            from integrated_cell.model_utils import sampleUniform as latentSample

        elif self.latent_distribution == 'gaussian':
            from integrated_cell.model_utils import sampleGaussian as latentSample

        self.latentSample = latentSample
        
        columns = ('epoch', 'iter', 'reconLoss',)
        print_str = '[%d][%d] reconLoss: %.6f'

        if self.n_classes > 0:
            columns += ('classLoss',)
            print_str += ' classLoss: %.6f'

        if self.n_ref > 0:
            columns += ('refLoss',)
            print_str += ' refLoss: %.6f'

        columns += ('minimaxEncDLoss', 'encDLoss', 'minimaxDecDLoss', 'decDLoss', 'time')
        print_str += ' mmEncD: %.6f encD: %.6f  mmDecD: %.6f decD: %.6f time: %.2f'

        self.logger = SimpleLogger(columns,  print_str)
        
    
            
            
    def load(self, save_dir):
        gpu_id = self.gpu_ids[0]
        
        if os.path.exists('{0}/enc.pth'.format(save_dir)):
            print('Loading from ' + save_dir)

            model_utils.load_state(self.enc, self.optEnc, '{0}/enc.pth'.format(save_dir), gpu_id)
            model_utils.load_state(self.dec, self.optDec, '{0}/dec.pth'.format(save_dir), gpu_id)
            model_utils.load_state(self.encD, self.optEncD, '{0}/encD.pth'.format(save_dir), gpu_id)
            model_utils.load_state(self.decD, self.optDecD, '{0}/decD.pth'.format(save_dir), gpu_id)


            self.logger = pickle.load(open( '{0}/logger.pkl'.format(save_dir), "rb" ))

    def save(self, save_dir):
#         for saving and loading see:
#         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        gpu_id = self.gpu_ids[0]

        model_utils.save_state(self.enc, self.optEnc, '{0}/enc.pth'.format(save_dir), gpu_id)
        model_utils.save_state(self.dec, self.optDec, '{0}/dec.pth'.format(save_dir), gpu_id)
        model_utils.save_state(self.encD, self.optEncD, '{0}/encD.pth'.format(save_dir), gpu_id)
        model_utils.save_state(self.decD, self.optDecD, '{0}/decD.pth'.format(save_dir), gpu_id)

        pickle.dump(self.logger, open('{0}/logger.pkl'.format(save_dir), 'wb'))



        
