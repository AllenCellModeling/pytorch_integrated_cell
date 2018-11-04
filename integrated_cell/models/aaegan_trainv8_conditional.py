import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

import importlib

from integrated_cell.model_utils import *
from integrated_cell.models import aaegan_trainv8 
from integrated_cell.utils import plots as plots

import integrated_cell.utils as utils



class Model(aaegan_trainv8.Model):
    def __init__(self, source_channels = [0,1,2], target_channels = [1], **kwargs):
        
        super(Model, self).__init__(**kwargs)
   
        self.target_channels = target_channels
        self.source_channels = source_channels

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
        
        xHat_full = x.clone()

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

        zAll = enc(x[:, self.source_channels])

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
        ### Train decD
        ##############

        yHat_xReal = decD(x)

        ### train with real
        errDecD_real = critDecD(yHat_xReal, y_xReal)

        ### train with fake, reconstructed
        
        xHat_full[:, self.target_channels] = xHat
        
        yHat_xFake = decD(xHat_full)
        errDecD_fake = critDecD(yHat_xFake, y_xFake)

        ### train with fake, sampled and decoded
        zAll[-1] = zReal

        xHat_full[:, self.target_channels] = dec(zAll)
        
        yHat_xFake2 = decD(xHat_full)
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
        zAll = enc(x[:, self.source_channels])

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
        reconLoss = critRecon(xHat, x[:, self.target_channels])
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
        
        xHat_full[:, self.target_channels] = xHat
        
        yHat_xFake = decD(xHat_full)
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

        # if self.n_ref > 0:
        #     zAll[c].data.normal_()


        #sample random positions in the localization space
        self.zReal.data.normal_()
        zAll[-1] = self.zReal

        xHat_full[:, self.target_channels] = dec(zAll)
        
        yHat_xFake2 = decD(xHat_full)
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

    def save_progress(self):
        gpu_id = self.gpu_ids[0]
        epoch = self.get_current_epoch()

        data_provider = self.data_provider
        enc = self.enc
        dec = self.dec

        enc.train(False)
        dec.train(False)

        ###############
        #TRAINING DATA
        ###############
        train_classes = data_provider.get_classes(np.arange(0, data_provider.get_n_dat('train', override=True)), 'train')
        _, train_inds = np.unique(train_classes.numpy(), return_index=True)

        x, classes, ref = data_provider.get_sample(train_inds,'train')
        x = x.cuda(gpu_id)
        classes = classes.cuda(gpu_id)
        ref = ref.cuda(gpu_id)
        
        xHat = x.clone()
        with torch.no_grad():            
            zAll = enc(x[:, self.source_channels])            
            zAll = self.setup_decoder_vars(zAll, classes, ref)
            xHat[:, self.target_channels] = dec(zAll)
            
        imgX = tensor2img(x.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgTrainOut = np.concatenate((imgX, imgXHat), 0)


        ###############
        #TESTING DATA
        ###############        
        test_classes = data_provider.get_classes(np.arange(0, data_provider.get_n_dat('test')), 'test')
        _, test_inds = np.unique(test_classes.numpy(), return_index=True)

        x, classes, ref = data_provider.get_sample(test_inds,'test')
        x = x.cuda(gpu_id)
        classes = classes.cuda(gpu_id)
        ref = ref.cuda(gpu_id)
        
        xHat = x.clone()
        with torch.no_grad():
            zAll = enc(x[:, self.source_channels])            
            zAll = self.setup_decoder_vars(zAll, classes, ref)
            xHat[:, self.target_channels] = dec(zAll)
            
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

        x_z = torch.zeros(x.shape).type_as(x)
        with torch.no_grad():
            x_z[:, self.target_channels] = dec(z)

        imgX = tensor2img(x.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgX_z = tensor2img(x_z.data.cpu())
        imgTestOut = np.concatenate((imgX, imgXHat, imgX_z), 0)

        imgOut = np.concatenate((imgTrainOut, imgTestOut))

        scipy.misc.imsave('{0}/progress_{1}.png'.format(self.save_dir, int(epoch)), imgOut)

        enc.train(True)
        dec.train(True)

        # pdb.set_trace()
        # zAll = torch.cat(zAll,0).cpu().numpy()

        embedding = torch.cat(self.zAll,0).cpu().numpy()

        pickle.dump(embedding, open('{0}/embedding_tmp.pkl'.format(self.save_dir), 'wb'))
        pickle.dump(self.logger, open('{0}/logger_tmp.pkl'.format(self.save_dir), 'wb'))

        ### History
        plots.history(self.logger, '{0}/history.png'.format(self.save_dir))

        ### Short History
        plots.short_history(self.logger, '{0}/history_short.png'.format(self.save_dir))

        ### Embedding figure
        plots.embeddings(embedding, '{0}/embedding.png'.format(self.save_dir))

        xHat = None
        x = None