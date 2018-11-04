import torch
import importlib
import torch.optim as optim
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.misc
import pickle
import importlib

import integrated_cell
import integrated_cell.utils
from integrated_cell import SimpleLogger
from integrated_cell import imgtoprojection
from integrated_cell import models



import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

import pdb

def init_opts(opt, opt_default):
    vars_default = vars(opt_default)
    for var in vars_default:
        if not hasattr(opt, var):
            setattr(opt, var, getattr(opt_default, var))
    return opt

def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                if gpu_id != -1:
                    var[key] = var[key].cuda(gpu_id)
                else:
                    var[key] = var[key].cpu()
            except:
                pass
    return var

def sampleUniform (batsize, nlatentdim):
    return torch.Tensor(batsize, nlatentdim).uniform_(-1, 1)

def sampleGaussian (batsize, nlatentdim):
    return torch.Tensor(batsize, nlatentdim).normal_()

def tensor2img(img):

    img = img.numpy()
    im_out = list()
    for i in range(0, img.shape[0]):
        im_out.append(img[i])

    img = np.concatenate(im_out, 2)

    if len(img.shape) == 3:
        img = np.expand_dims(img, 3)

    colormap = 'hsv'

    colors = plt.get_cmap(colormap)(np.linspace(0, 1, img.shape[0]+1))

    # img = np.swapaxes(img, 2,3)
    img = imgtoprojection(np.swapaxes(img, 1, 3), colors = colors, global_adjust=True)
    img = np.swapaxes(img, 0, 2)

    return img

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_embeddings(embeddings_path, enc=None, dp=None, opt=None):

    if os.path.exists(embeddings_path):
        embeddings = torch.load(embeddings_path)
    else:
        embeddings = get_latent_embeddings(enc, dp, opt)
        torch.save(embeddings, embeddings_path)

    return embeddings

def get_latent_embeddings(enc, dp, opt):
    enc.eval()
    gpu_id = opt.gpu_ids[0]

    modes = ('test', 'train')

    embedding = dict()

    for mode in modes:
        ndat = dp.get_n_dat(mode)
        embeddings = torch.zeros(ndat, opt.nlatentdim)

        inds = list(range(0,ndat))
        data_iter = [inds[i:i+opt.batch_size] for i in range(0, len(inds), opt.batch_size)]

        for i in range(0, len(data_iter)):
            print(str(i) + '/' + str(len(data_iter)))
            x = dp.get_images(data_iter[i], mode).cuda(gpu_id)
            
            with torch.no_grad():
                zAll = enc(x)

            embeddings.index_copy_(0, torch.LongTensor(data_iter[i]), zAll[-1].data[:].cpu())

        embedding[mode] = embeddings

    return embedding

def load_data_provider(data_path, batch_size, im_dir, dp_module, n_dat = -1, **kwargs_dp):
    DP = importlib.import_module("integrated_cell.data_providers." + dp_module)

    if os.path.exists(data_path):
        dp = torch.load(data_path)
        dp.image_parent = im_dir
    else:
        dp = DP.DataProvider(im_dir, batch_size = batch_size, n_dat = n_dat, **kwargs_dp)
        torch.save(dp, data_path)

    dp.batch_size = batch_size
    
    if n_dat > 0:
        dp.n_dat['train'] = n_dat
    
    return dp


def fix_data_paths(parent_dir, new_im_dir = None, data_save_path = None):
    #this will only work with the h5 dataprovider

    from shutil import copyfile

    def rename_opt_path(opt_dir):

        pkl_path = '{0}/opt.pkl'.format(opt_dir)
        pkl_path_bak = '{0}/opt.pkl.bak'.format(opt_dir)

        copyfile(pkl_path, pkl_path_bak)

        opt = pickle.load(open(pkl_path, "rb" ))
        opt.imdir = new_im_dir
        opt.data_save_path = data_save_path
        opt.save_parent = parent_dir
        opt.save_dir = opt_dir
        pickle.dump(opt, open(pkl_path, 'wb'))

    if data_save_path is None:
        data_save_path = '{0}/data.pyt'.format(parent_dir)

    ref_dir = parent_dir + os.sep + 'ref_model'
    rename_opt_path(ref_dir)

    struct_dir = parent_dir + os.sep + 'struct_model'
    rename_opt_path(struct_dir)

    opt = pickle.load(open('{0}/opt.pkl'.format(ref_dir), "rb" ))

    if new_im_dir is None:
        new_im_dir = opt.imdir

    copyfile(opt.data_save_path, opt.data_save_path + '.bak')

    dp = load_data_provider(opt.data_save_path, opt.imdir, opt.dataProvider)
    dp.image_parent = new_im_dir
    torch.save(dp, opt.data_save_path)


    pass


def load_state(model, optimizer, path, gpu_id):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model'])
    model.cuda(gpu_id)

    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)

def save_state(model, optimizer, path, gpu_id):

    model = model.cpu()
    optimizer.state = set_gpu_recursive(optimizer.state, -1)

    checkpoint = {'model': model.state_dict(),
              'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, path)

    model = model.cuda(gpu_id)
    optimizer.state = set_gpu_recursive(optimizer.state, gpu_id)

def load_network_from_path(net, net_optim, save_path, gpu_id):

    if os.path.exists(save_path):
        print('Loading from ' + save_path)

        model_utils.load_state(net, net_optim, save_path, gpu_id)