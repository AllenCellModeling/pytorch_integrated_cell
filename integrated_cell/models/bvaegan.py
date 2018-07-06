import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


from integrated_cell.model_utils import *
from integrated_cell.utils import plots as plots

# This is the trainer for the Beta-VAE

from integrated_cell.models import bvae
from integrated_cell.models import base_model 


class Model(base_model.Model):
    def __init__(self, data_provider, n_channels, batch_size, n_latent_dim, n_classes, n_ref, gpu_ids, beta = 1, beta_step = None, objective = 'H'):
        super(Model, self).__init__(data_provider, n_channels, batch_size, n_latent_dim, n_classes, n_ref, gpu_ids)
        
        if beta_step is None:
            self.beta = beta
        else:
            self.beta = 0
            self.beta_step = beta_step
            self.beta_max = beta
            
        self.objective = objective


    def iteration(self,
                  enc, dec, decD,
                  optEnc, optDec, optDecD,
                  critRecon, critZClass, critZRef, critDecD,
                  data_provider, opt):
        gpu_id = self.gpu_ids[0]

        #do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)
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

        for p in decD.parameters():
            p.requires_grad = True

        for p in enc.parameters():
            p.requires_grad = False

        for p in dec.parameters():
            p.requires_grad = False

        zAll = enc(x)

        for i in range(len(zAll)-1):
                zAll[i].detach_()

        for var in zAll[-1]:
            var.detach_()

        zAll[-1] = bvae.reparameterize(zAll[-1][0], zAll[-1][1])

        xHat = dec(zAll)

        self.zReal.data.normal_()
        zReal = self.zReal
        # zReal = Variable(opt.latentSample(self.batch_size, self.n_latent_dim).cuda(gpu_id))
        zFake = zAll[-1]

        optEnc.zero_grad()
        optDec.zero_grad()
        optDecD.zero_grad()

        ##############
        ### Train decD
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

        errDecD_real = None
        errDecD_fake = None
        errDecD_fake2 = None

        for p in enc.parameters():
            p.requires_grad = True

        for p in dec.parameters():
            p.requires_grad = True

        for p in decD.parameters():
            p.requires_grad = False

        optEnc.zero_grad()
        optDec.zero_grad()
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
            classLoss.backward(retain_graph=True)
            classLoss = classLoss.data[0]
            c += 1

        ### Update the reference shape discriminator
        if self.n_ref > 0:
            refLoss = critZRef(zAll[c], ref)
            refLoss.backward(retain_graph=True)
            refLoss = refLoss.data[0]
            c += 1


        total_kld, dimension_wise_kld, mean_kld = bvae.kl_divergence(zAll[c][0], zAll[c][1])
        kld_loss = total_kld.data[0]
        minimaxEncDLoss = 0

        zAll[c] = bvae.reparameterize(zAll[c][0], zAll[c][1])

        xHat = dec(zAll)

        ### Update the image reconstruction
        reconLoss = critRecon(xHat, x)

        if self.objective == 'H':
            beta_vae_loss = reconLoss + self.beta*total_kld
        elif self.objective == 'B':
            C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
            beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()


        beta_vae_loss.backward(retain_graph=True)
        reconLoss = reconLoss.data[0]

        optEnc.step()

        for p in enc.parameters():
            p.requires_grad = False

        ### update wrt decD(dec(enc(X)))
        yHat_xFake = decD(xHat)
        minimaxDecDLoss = critDecD(yHat_xFake, y_xReal)
        (minimaxDecDLoss.mul(opt.lambdaDecD).div(2)).backward(retain_graph=True)
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
        (minimaxDecDLoss2.mul(opt.lambdaDecD).div(2)).backward(retain_graph=True)
        minimaxDecDLoss2 = minimaxDecDLoss2.data[0]
        yHat_xFake2 = None

        minimaxDecLoss = (minimaxDecDLoss+minimaxDecDLoss2)/2

        optDec.step()

        errors = (reconLoss,)
        if self.n_classes > 0:
            errors += (classLoss,)

        if self.n_ref > 0:
            errors += (refLoss,)

        errors += (kld_loss, minimaxDecLoss, decDLoss)

        
        if self.beta_step is not None:
            self.beta += self.beta_step
            
            if self.beta > self.beta_max:
                self.beta = self.beta_max
        
        return errors, zFake.data


    def load(self, model_name, opt):

        model_provider = importlib.import_module("integrated_cell.networks." + model_name)

        enc = model_provider.Enc(self.n_latent_dim, self.n_classes, self.n_ref, self.n_channels, self.gpu_ids, **opt.kwargs_enc)
        dec = model_provider.Dec(self.n_latent_dim, self.n_classes, self.n_ref, self.n_channels, self.gpu_ids, **opt.kwargs_dec)
        decD = model_provider.DecD(self.n_classes+1, self.n_channels, self.gpu_ids, **opt.kwargs_decD)

        enc.apply(weights_init)
        dec.apply(weights_init)
        decD.apply(weights_init)

        gpu_id = self.gpu_ids[0]

        enc.cuda(gpu_id)
        dec.cuda(gpu_id)
        decD.cuda(gpu_id)

        if opt.optimizer == 'RMSprop':
            optEnc = optim.RMSprop(enc.parameters(), lr=opt.lrEnc)
            optDec = optim.RMSprop(dec.parameters(), lr=opt.lrDec)
            optDecD = optim.RMSprop(decD.parameters(), lr=opt.lrDecD)
        elif opt.optimizer == 'adam':

            optEnc = optim.Adam(enc.parameters(), lr=opt.lrEnc, **opt.kwargs_optim)
            optDec = optim.Adam(dec.parameters(), lr=opt.lrDec, **opt.kwargs_optim)
            optDecD = optim.Adam(decD.parameters(), lr=opt.lrDecD, **opt.kwargs_optim)

        columns = ('epoch', 'iter', 'reconLoss',)
        print_str = '[%d][%d] reconLoss: %.6f'

        if self.n_classes > 0:
            columns += ('classLoss',)
            print_str += ' classLoss: %.6f'

        if self.n_ref > 0:
            columns += ('refLoss',)
            print_str += ' refLoss: %.6f'

        columns += ('kldLoss', 'minimaxDecDLoss', 'decDLoss', 'time')
        print_str += ' kld: %.6f mmDecD: %.6f decD: %.6f time: %.2f'

        logger = SimpleLogger(columns,  print_str)

        if os.path.exists('{0}/enc.pth'.format(opt.save_dir)):
            print('Loading from ' + opt.save_dir)

            load_state(enc, optEnc, '{0}/enc.pth'.format(opt.save_dir), gpu_id)
            load_state(dec, optDec, '{0}/dec.pth'.format(opt.save_dir), gpu_id)
            load_state(decD, optDecD, '{0}/decD.pth'.format(opt.save_dir), gpu_id)

            logger = pickle.load(open( '{0}/logger.pkl'.format(opt.save_dir), "rb" ))

            this_epoch = max(logger.log['epoch']) + 1
            iteration = max(logger.log['iter'])

        models = {'enc': enc, 'dec': dec, 'decD': decD}

        optimizers = dict()
        optimizers['optEnc'] = optEnc
        optimizers['optDec'] = optDec
        optimizers['optDecD'] = optDecD

        criterions = dict()
        criterions['critRecon'] = eval('nn.' + opt.critRecon + '(size_average=False)')
        criterions['critZClass'] = nn.NLLLoss()
        criterions['critZRef'] = nn.MSELoss()


        if self.n_classes > 0:
            criterions['critDecD'] = nn.CrossEntropyLoss()
        else:
            criterions['critDecD'] = nn.BCEWithLogitsLoss()

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

    def save(self, enc, dec, decD,
                   optEnc, optDec, optDecD,
                   logger, zAll, opt):
    #         for saving and loading see:
    #         https://discuss.pytorch.org/t/how-to-save-load-torch-models/718

        gpu_id = self.gpu_ids[0]

        save_state(enc, optEnc, '{0}/enc.pth'.format(opt.save_dir), gpu_id)
        save_state(dec, optDec, '{0}/dec.pth'.format(opt.save_dir), gpu_id)
        save_state(decD, optDecD, '{0}/decD.pth'.format(opt.save_dir), gpu_id)

        pickle.dump(zAll, open('{0}/embedding.pkl'.format(opt.save_dir), 'wb'))
        pickle.dump(logger, open('{0}/logger.pkl'.format(opt.save_dir), 'wb'))

