import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


class trainer(object):
    def __init__(self, dp, opt):
        
        gpu_id = opt.gpu_ids[0]
        
        self.x = Variable(dp.get_images(range(0, opt.batch_size),'train').cuda(gpu_id))
            
        if opt.nRef > 0:
            self.ref = Variable(dp.get_ref(range(0, opt.batch_size), train_or_test='train').type_as(self.x.data).cuda(gpu_id))
        else:
            self.ref = None
        
        self.zReal = Variable(torch.Tensor(opt.batch_size, opt.nlatentdim).type_as(self.x.data).cuda(gpu_id))
        
        
        #zReal is nClasses + 1
        if opt.nClasses == 0:
            self.y_zReal = Variable(torch.Tensor(opt.batch_size, 1).type_as(self.x.data).cuda(gpu_id))
            self.y_zReal.data.fill_(1)
            
            self.y_zFake = Variable(torch.Tensor(opt.batch_size, 1).type_as(self.x.data).cuda(gpu_id))
            self.y_zFake.data.fill_(0)
            #zFake is nClasses (either 0 (no classification), or opt.nClasses (multi class))
            
            self.y_xReal = self.y_zReal
            self.y_xFake = self.y_zFake
            
            
        else:    
            self.y_zReal = Variable(torch.LongTensor(opt.batch_size).cuda(gpu_id))
            self.y_zReal.data.fill_(opt.nClasses)
            
            self.y_zFake = Variable(torch.LongTensor(opt.batch_size).cuda(gpu_id))
            #dont do anything with y_zFake since it gets filled from the dataprovider
            #self.y_zFake
            

            self.y_xFake = Variable(torch.LongTensor(opt.batch_size).cuda(gpu_id))
            self.y_xFake.data.fill_(opt.nClasses)
            
            self.y_xReal = Variable(torch.LongTensor(opt.batch_size).cuda(gpu_id))
            #dont do anything with y_xReal since it gets filled from the dataprovider
            #self.y_xReal
    
            self.classes = Variable(torch.LongTensor(opt.batch_size).cuda(gpu_id))
        
    def iteration(self, 
                  enc, dec, encD, decD, 
                  optEnc, optDec, optEncD, optDecD, 
                  critRecon, critZClass, critZRef, critEncD, critDecD,
                  dataProvider, opt):
        gpu_id = opt.gpu_ids[0]

        #do this just incase anything upstream changes these values
        enc.train(True)
        dec.train(True)
        encD.train(True)
        decD.train(True)

        ###update the discriminator
        #maximize log(AdvZ(z)) + log(1 - AdvZ(Enc(x)))

        rand_inds_encD = np.random.permutation(opt.ndat)
        inds = rand_inds_encD[0:opt.batch_size]
        
        self.x.data.copy_(dataProvider.get_images(inds,'train'))
        x = self.x

        y_xFake = self.y_xFake
        y_zReal = self.y_zReal
        
        if opt.nClasses == 0:
            y_xReal = self.y_xReal
#             y_xFake = self.y_xFake
            
#             y_zReal = self.y_zReal
            y_zFake = self.y_zFake
        else:
            self.classes.data.copy_(dataProvider.get_classes(inds,'train'))
            classes = self.classes
            
            y_xReal = classes
            y_zFake = classes

        if opt.nRef > 0:
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
        if opt.nClasses > 0:
            classLoss = critZClass(zAll[c], classes)
            classLoss.backward(retain_graph=True)        
            classLoss = classLoss.data[0]
            c += 1

        ### Update the reference shape discriminator
        if opt.nRef > 0:
            refLoss = critZRef(zAll[c], ref)
            refLoss.backward(retain_graph=True)   
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
        (minimaxEncDLoss.mul(opt.encDRatio)).backward(retain_graph=True)
        minimaxEncDLoss = minimaxEncDLoss.data[0]
        
        optEnc.step()

        for p in enc.parameters():
            p.requires_grad = False

        ### update wrt decD(dec(enc(X)))
        yHat_xFake = decD(xHat)
        minimaxDecDLoss = critDecD(yHat_xFake, y_xReal)
        (minimaxDecDLoss.mul(opt.decDRatio).div(2)).backward(retain_graph=True)
        minimaxDecDLoss = minimaxDecDLoss.data[0]
        yHat_xFake = None
        
        ### update wrt decD(dec(Z))
        
        c = 0  
        #if we have classes, create random classes, generate images of random classes
        if opt.nClasses > 0:
            shuffle_inds = np.arange(0, zAll[0].size(0))
            
            classes_one_hot = Variable((dataProvider.get_classes(inds,'train', 'one hot') - 1) * 25).type_as(zAll[c].data).cuda(opt.gpu_ids[0]) 
            
            np.random.shuffle(shuffle_inds)
            zAll[c] = classes_one_hot[shuffle_inds,:]
            y_xReal = y_xReal[torch.LongTensor(shuffle_inds).cuda(opt.gpu_ids[0])]
            
            c +=1
            
        if opt.nRef > 0:
            zAll[c].data.normal_()

        
        #sample random positions in the localization space
        self.zReal.data.normal_()
        zAll[-1] = self.zReal
        
        xHat = dec(zAll)

        yHat_xFake2 = decD(xHat)
        minimaxDecDLoss2 = critDecD(yHat_xFake2, y_xReal)
        (minimaxDecDLoss2.mul(opt.decDRatio).div(2)).backward(retain_graph=True)
        minimaxDecDLoss2 = minimaxDecDLoss2.data[0]
        yHat_xFake2 = None
        
        minimaxDecLoss = (minimaxDecDLoss+minimaxDecDLoss2)/2
        
        optDec.step()
        
        errors = (reconLoss,)
        if opt.nClasses > 0:
            errors += (classLoss,)

        if opt.nRef > 0:
            errors += (refLoss,)

        errors += (minimaxEncDLoss, encDLoss, minimaxDecLoss, decDLoss)

        return errors, zFake.data