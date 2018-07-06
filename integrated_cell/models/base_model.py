import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb


from integrated_cell.model_utils import *
from integrated_cell.utils import plots as plots

# This is the base class for trainers

class Model(object):
    def __init__(self, data_provider, n_channels, batch_size, n_latent_dim, n_classes, n_ref, gpu_ids):

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

        raise NotImplementedError


    def load(self, model_name, opt):

        raise NotImplementedError

    def save(self, enc, dec, decD,
                   optEnc, optDec, optDecD,
                   logger, zAll, opt):
        
        raise NotImplementedError


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
