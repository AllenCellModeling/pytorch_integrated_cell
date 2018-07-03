import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


from integrated_cell.model_utils import *
from integrated_cell.utils import plots as plots

# This is the trainer for the Beta-VAE

def reparameterize(mu, log_var):
    std = log_var.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

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


class Model(object):
    def __init__(self, data_provider, n_channels, batch_size, n_latent_dim, n_classes, n_ref, gpu_ids):

        self.objective = 'H'
        self.beta = 1

        self.batch_size = batch_size

        self.n_channels = n_channels

        self.n_ref = n_ref
        self.n_latent_dim = n_latent_dim
        self.n_classes = n_classes
        self.gpu_ids = gpu_ids

        gpu_id = self.gpu_ids[0]

        self.x = Variable(data_provider.get_images(range(0, self.batch_size),'train').cuda(gpu_id))

        if self.n_ref > 0:
            self.ref = Variable(data_provider.get_ref(range(0, self.batch_size), train_or_test='train').type_as(self.x.data).cuda(gpu_id))
        else:
            self.ref = None

        self.zReal = Variable(torch.Tensor(self.batch_size, self.n_latent_dim).type_as(self.x.data).cuda(gpu_id))


        #zReal is nClasses + 1
        if self.n_classes == 0:
            self.y_zReal = Variable(torch.Tensor(self.batch_size, 1).type_as(self.x.data).cuda(gpu_id))
            self.y_zReal.data.fill_(1)

            self.y_zFake = Variable(torch.Tensor(self.batch_size, 1).type_as(self.x.data).cuda(gpu_id))
            self.y_zFake.data.fill_(0)
            #zFake is nClasses (either 0 (no classification), or self.n_classes (multi class))

            self.y_xReal = self.y_zReal
            self.y_xFake = self.y_zFake


        else:
            self.y_zReal = Variable(torch.LongTensor(self.batch_size).cuda(gpu_id))
            self.y_zReal.data.fill_(self.n_classes)

            self.y_zFake = Variable(torch.LongTensor(self.batch_size).cuda(gpu_id))
            #dont do anything with y_zFake since it gets filled from the dataprovider
            #self.y_zFake


            self.y_xFake = Variable(torch.LongTensor(self.batch_size).cuda(gpu_id))
            self.y_xFake.data.fill_(self.n_classes)

            self.y_xReal = Variable(torch.LongTensor(self.batch_size).cuda(gpu_id))
            #dont do anything with y_xReal since it gets filled from the dataprovider
            #self.y_xReal

            self.classes = Variable(torch.LongTensor(self.batch_size).cuda(gpu_id))


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

        zAll[-1] = reparameterize(zAll[-1][0], zAll[-1][1])

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


        total_kld, dimension_wise_kld, mean_kld = kl_divergence(zAll[c][0], zAll[c][1])
        kld_loss = total_kld.data[0]
        minimaxEncDLoss = 0

        zAll[c] = reparameterize(zAll[c][0], zAll[c][1])

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
        criterions['critRecon'] = eval('nn.' + opt.critRecon + '()')
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


    def save_progress(self, enc, dec, data_provider, logger, embedding, opt):
        gpu_id = self.gpu_ids[0]

        epoch = max(logger.log['epoch'])

        enc.train(False)
        dec.train(False)

    #     pdb.set_trace()
        train_classes = data_provider.get_classes(np.arange(0, data_provider.get_n_dat('train')), 'train')
        _, train_inds = np.unique(train_classes.numpy(), return_index=True)

        x = Variable(data_provider.get_images(train_inds,'train').cuda(gpu_id), volatile=True)

        xHat = dec(enc(x))
        imgX = tensor2img(x.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgTrainOut = np.concatenate((imgX, imgXHat), 0)

        test_classes = data_provider.get_classes(np.arange(0, data_provider.get_n_dat('test')), 'test')
        _, test_inds = np.unique(test_classes.numpy(), return_index=True)

        x = Variable(data_provider.get_images(test_inds,'test').cuda(gpu_id), volatile=True)
        xHat = dec(enc(x))

        z = list()
        if self.n_classes > 0:
            class_var = Variable(torch.Tensor(data_provider.get_classes(test_inds, 'test', 'one_hot').float()).cuda(gpu_id), volatile=True)
            class_var = (class_var-1) * 25
            z.append(class_var)

        if self.n_ref > 0:
            ref_var = Variable(torch.Tensor(data_provider.get_n_classes(), self.n_ref).normal_(0,1).cuda(gpu_id), volatile=True)
            z.append(ref_var)

        loc_var = Variable(torch.Tensor(data_provider.get_n_classes(), self.n_latent_dim).normal_(0,1).cuda(gpu_id), volatile=True)
        z.append(loc_var)

        x_z = dec(z)

        imgX = tensor2img(x.data.cpu())
        imgXHat = tensor2img(xHat.data.cpu())
        imgX_z = tensor2img(x_z.data.cpu())
        imgTestOut = np.concatenate((imgX, imgXHat, imgX_z), 0)

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
