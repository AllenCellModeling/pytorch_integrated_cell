import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

import importlib

from integrated_cell.model_utils import *
from integrated_cell.models import base_model 

class Model(base_model.Model):
    def __init__(self, data_provider, 
                 n_channels, 
                 batch_size, 
                 n_latent_dim, 
                 n_classes, 
                 n_ref, 
                 gpu_ids, 
                 lambda_encD_loss = 5E-3,
                 lambda_decD_loss = 1E-4,
                 lambda_ref_loss = 1,
                 lambda_class_loss = 1,
                 size_average_losses = False,
                 provide_decoder_vars = False,
                ):
        
        super(Model, self).__init__(data_provider, n_channels, batch_size, n_latent_dim, n_classes, n_ref, gpu_ids)
 
        self.provide_decoder_vars = provide_decoder_vars
    
        self.size_average_losses = size_average_losses
    
        self.lambda_encD_loss = lambda_encD_loss
        self.lambda_decD_loss = lambda_decD_loss
        self.lambda_ref_loss = lambda_ref_loss
        self.lambda_class_loss = lambda_class_loss
        
        self.size_average_losses = size_average_losses

    def iteration(self,
                  enc, dec, encD, decD,
                  optEnc, optDec, optEncD, optDecD,
                  critRecon, critZClass, critZRef, critEncD, critDecD,
                  data_provider, opt):
        gpu_id = self.gpu_ids[0]

        #do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)
        encD.train(True)
        decD.train(True)

        ###update the discriminator
        #maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))

        rand_inds_encD = np.random.permutation(opt.ndat)
        inds = rand_inds_encD[0:self.batch_size]

        self.x.data.copy_(data_provider.get_images(inds,'train'))
        x = self.x

        y_xFake = self.y_xFake
        y_zReal = self.y_zReal

        if self.n_classes == 0:
            y_xReal = self.y_xReal
            y_zFake = self.y_zFake
        else:
            self.classes.data.copy_(data_provider.get_classes(inds,'train'))
            classes = self.classes

            y_xReal = classes
            y_zFake = classes

        if self.n_ref > 0:
            self.ref.data.copy_(data_provider.get_ref(inds,'train'))
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

        for var in zAll:
            var.detach_()

        xHat = dec(zAll)

        self.zReal.data.normal_()
        zReal = self.zReal
        # zReal = Variable(opt.latentSample(self.batch_size, opt.nlatentdim).cuda(gpu_id))
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
            c += 1

        ### Update the reference shape discriminator
        if self.n_ref > 0:
            refLoss = critZRef(zAll[c], ref)
            refLoss.mul(self.lambda_ref_loss).backward(retain_graph=True)
            refLoss = refLoss.data[0]
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

            classes_one_hot = Variable((data_provider.get_classes(inds,'train', 'one hot') - 1) * 25).type_as(zAll[c].data).cuda(self.gpu_ids[0])

            np.random.shuffle(shuffle_inds)
            zAll[c] = classes_one_hot[shuffle_inds,:]
            y_xReal = y_xReal[torch.LongTensor(shuffle_inds).cuda(self.gpu_ids[0])]

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

    def load(self, model_name, opt):

        model_provider = importlib.import_module("integrated_cell.networks." + model_name)

        enc = model_provider.Enc(self.n_latent_dim, self.n_classes, self.n_ref, self.n_channels, self.gpu_ids, **opt.kwargs_enc)
        dec = model_provider.Dec(self.n_latent_dim, self.n_classes, self.n_ref, self.n_channels, self.gpu_ids, **opt.kwargs_dec)
        encD = model_provider.EncD(self.n_latent_dim, self.n_classes+1, self.gpu_ids, **opt.kwargs_encD)
        decD = model_provider.DecD(self.n_classes+1, self.n_channels, self.gpu_ids, **opt.kwargs_decD)

        enc.apply(weights_init)
        dec.apply(weights_init)
        encD.apply(weights_init)
        decD.apply(weights_init)

        gpu_id = self.gpu_ids[0]

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
            optEnc = optim.Adam(enc.parameters(), lr=opt.lrEnc, **opt.kwargs_optim)
            optDec = optim.Adam(dec.parameters(), lr=opt.lrDec, **opt.kwargs_optim)
            optEncD = optim.Adam(encD.parameters(), lr=opt.lrEncD, **opt.kwargs_optim)
            optDecD = optim.Adam(decD.parameters(), lr=opt.lrDecD, **opt.kwargs_optim)

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

        logger = SimpleLogger(columns,  print_str)

        if os.path.exists('{0}/enc.pth'.format(opt.save_dir)):
            print('Loading from ' + opt.save_dir)

            load_state(enc, optEnc, '{0}/enc.pth'.format(opt.save_dir), gpu_id)
            load_state(dec, optDec, '{0}/dec.pth'.format(opt.save_dir), gpu_id)
            load_state(encD, optEncD, '{0}/encD.pth'.format(opt.save_dir), gpu_id)
            load_state(decD, optDecD, '{0}/decD.pth'.format(opt.save_dir), gpu_id)

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
        criterions['critRecon'] = eval('nn.' + opt.critRecon + '(size_average=' + str(bool(self.size_average_losses)) + ')')
        criterions['critZClass'] = nn.NLLLoss(size_average=self.size_average_losses)
        criterions['critZRef'] = nn.MSELoss(size_average=self.size_average_losses)


        if self.n_classes > 0:
            criterions['critDecD'] = nn.CrossEntropyLoss(size_average=self.size_average_losses)
            criterions['critEncD'] = nn.CrossEntropyLoss(size_average=self.size_average_losses)
        else:
            criterions['critEncD'] = nn.BCEWithLogitsLoss(size_average=self.size_average_losses)
            criterions['critDecD'] = nn.BCEWithLogitsLoss(size_average=self.size_average_losses)

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
    
    def save(self, enc, dec, encD, decD,
                       optEnc, optDec, optEncD, optDecD,
                       logger, zAll, opt):
#         for saving and loading see:
#         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        gpu_id = self.gpu_ids[0]


        save_state(enc, optEnc, '{0}/enc.pth'.format(opt.save_dir), gpu_id)
        save_state(dec, optDec, '{0}/dec.pth'.format(opt.save_dir), gpu_id)
        save_state(encD, optEncD, '{0}/encD.pth'.format(opt.save_dir), gpu_id)
        save_state(decD, optDecD, '{0}/decD.pth'.format(opt.save_dir), gpu_id)

        pickle.dump(zAll, open('{0}/embedding.pkl'.format(opt.save_dir), 'wb'))
        pickle.dump(logger, open('{0}/logger.pkl'.format(opt.save_dir), 'wb'))



        
