import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

from integrated_cell.models import aaegan_trainv8 


class Model(aaegan_trainv8.Model):
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
        
        super(Model, self).__init__(data_provider, 
                 n_channels, 
                 batch_size, 
                 n_latent_dim, 
                 n_classes, 
                 n_ref, 
                 gpu_ids, 
                 lambda_encD_loss,
                 lambda_decD_loss,
                 lambda_ref_loss,
                 lambda_class_loss,
                 size_average_losses,
                 provide_decoder_vars,
                )
        
    def gradient_penalty(self, xreal, xfake, netD, lamba):
        
        dims = torch.Tensor([xreal.shape])[0].int()
        dims[1:] = 1
        
        alpha = torch.rand(tuple(dims)).type_as(xreal).expand_as(xreal)        
        # alpha.data.uniform_()

        
        interp_points = (alpha * xreal.data + (1 - alpha) * xfake.data)
        
        interp_points.requires_grad = True
        
        errD_interp_vec = netD(interp_points)
        errD_gradient, = torch.autograd.grad(errD_interp_vec.sum(), interp_points,
                                             create_graph=True, 
                                             retain_graph=True, 
                                             only_inputs=True)
        
        lip_est = (errD_gradient**2).view(self.batch_size,-1).sum(1)
        lip_loss = lamba*((1.0-lip_est)**2).mean(0).view(1)
        
        return lip_loss   
        
    def wasserstein_loss(self, outputs, targets):
        
        targets = targets.type_as(outputs)
        
        u_targets = torch.unique(targets.cpu()).type_as(outputs)
        
        loss = torch.Tensor([0]).type_as(outputs)
        
        for target in u_targets:
            target_loss = torch.mean(outputs[targets == target], 0)
            
            if self.nClasses == 0:
                if target == 0:
                    target_loss = -target_loss
            else:
                target_mask = torch.ones(opt.nClasses).type_as(outputs)
                target_mask[target] = 0
                
                target_loss = torch.mean(target_loss[target_mask]) - target_loss[target]
                
        loss = loss + target_loss
        
        return loss
        
        
    def iteration(self, 
                  enc, dec, encD, decD, 
                  optEnc, optDec, optEncD, optDecD, 
                  critRecon, critZClass, critZRef, critEncD, critDecD,
                  dataProvider, opt):
        gpu_id = opt.gpu_ids[0]

        
        critEncD = self.wasserstein_loss
        critDecD = self.wasserstein_loss
        
        #do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)
        encD.train(True)
        decD.train(True)

        ###update the discriminator
        #maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))

        rand_inds_encD = np.random.permutation(opt.ndat)
        inds = rand_inds_encD[0:self.batch_size]
        
        self.x.data.copy_(dataProvider.get_images(inds,'train'))
        x = self.x

        y_xFake = self.y_xFake
        y_zReal = self.y_zReal
        
        if self.n_classes == 0:
            y_xReal = self.y_xReal
#             y_xFake = self.y_xFake
            
#             y_zReal = self.y_zReal
            y_zFake = self.y_zFake
        else:
            self.classes.data.copy_(dataProvider.get_classes(inds,'train'))
            classes = self.classes
            
            y_xReal = classes
            y_zFake = classes

        if self.n_ref > 0:
            self.ref.data.copy_(dataProvider.get_ref(inds,'train'))
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
        # zReal = Variable(opt.latentSample(opt.batch_size, opt.nlatentdim).cuda(gpu_id))
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
        
        if self.gp_loss_lambda_encD > 0:
            encD_gp_loss = self.gradient_penalty(zReal, zFake, encD, self.gp_loss_lambda_encD)
            encD_gp_loss.backward(retain_graph=True)
        
        optEncD.step()
        
        encDLoss = encDLoss.data[0]

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
        xHat2 = dec(zAll)
        
        yHat_xFake2 = decD(xHat2)
        errDecD_fake2 = critDecD(yHat_xFake2, y_xFake)

        decDLoss = (errDecD_real + (errDecD_fake + errDecD_fake2)/2)/2
        decDLoss.backward(retain_graph=True)
        
        if self.gp_loss_lambda_decD > 0:
            decD_gp_loss = self.gradient_penalty(x, xHat, decD, self.gp_loss_lambda_decD)
            decD_gp_loss.backward(retain_graph=True)
            
            decD_gp_loss2 = self.gradient_penalty(x, xHat2, decD, self.gp_loss_lambda_decD)
            decD_gp_loss2.backward(retain_graph=True)
        
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
        if self.nClasses > 0:
            classLoss = critZClass(zAll[c], classes)
            classLoss.mul(self.lambda_class_loss).backward(retain_graph=True)      
            classLoss = classLoss.data[0]
            c += 1

        ### Update the reference shape discriminator
        if self.nRef > 0:
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
            
            classes_one_hot = Variable((dataProvider.get_classes(inds,'train', 'one hot') - 1) * 25).type_as(zAll[c].data).cuda(opt.gpu_ids[0]) 
            
            np.random.shuffle(shuffle_inds)
            zAll[c] = classes_one_hot[shuffle_inds,:]
            y_xReal = y_xReal[torch.LongTensor(shuffle_inds).cuda(opt.gpu_ids[0])]
            
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