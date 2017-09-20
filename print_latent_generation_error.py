#######    
### This function prints off the most likely predicted 
### channels for each of the cells in our dataset
#######

#######    
### Load the Model Parts
#######

import argparse

import importlib
import numpy as np

import os
import pickle

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils

#have to do this import to be able to use pyplot in the docker image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from IPython import display
import time
from model_utils import set_gpu_recursive, load_model, save_state, save_progress, get_latent_embeddings, maybe_save

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from tqdm import tqdm

from corr_stats import pearsonr, corrcoef

import pdb

parent_dir = './test_aaegan/aaegan3Dv5_128D'

model_dir = parent_dir + os.sep + 'struct_model' 

# logger_file = '{0}/logger_tmp.pkl'.format(model_dir)
opt = pickle.load( open( '{0}/opt.pkl'.format(model_dir), "rb" ) )

print(opt)

DP = importlib.import_module("data_providers." + opt.dataProvider)
model_provider = importlib.import_module("models." + opt.model_name)
train_module = importlib.import_module("train_modules." + opt.train_module)

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)
    
if opt.nepochs_pt2 == -1:
    opt.nepochs_pt2 = opt.nepochs
    
# opts = {}
# opts['verbose'] = True
# opts['pattern'] = '*.tif_flat.png'
# opts['out_size'] = [opt.imsize, opt.imsize]

data_path = './data_{0}x{1}.pyt'.format(str(opt.imsize), str(opt.imsize))
if os.path.exists(data_path):
    dp = torch.load(data_path)
else:
    dp = DP.DataProvider(opt.imdir)
    torch.save(dp, data_path)
    
dp.opts['dtype'] = 'float'
    
if opt.ndat == -1:
    opt.ndat = dp.get_n_dat('train')    

iters_per_epoch = np.ceil(opt.ndat/opt.batch_size)    
            
#######    
### Load REFERENCE MODEL
#######

embeddings_path = opt.save_parent + os.sep + 'ref_model' + os.sep + 'embeddings.pkl'
if os.path.exists(embeddings_path):
    embeddings = torch.load(embeddings_path)
else:
    embeddings = get_latent_embeddings(models['enc'], dp, opt)
    torch.save(embeddings, embeddings_path)

models = None
optimizers = None
    
def get_ref(self, inds, train_or_test='train'):
    inds = torch.LongTensor(inds)
    return self.embeddings[train_or_test][inds]

dp.embeddings = embeddings

# do this thing to bind the get_ref method to the dataprovider object
import types  
dp.get_ref = types.MethodType(get_ref, dp)
            

opt.channelInds = [0, 1, 2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = dp.get_n_classes()
opt.nRef = opt.nlatentdim

try:
    train_module = None
    train_module = importlib.import_module("train_modules." + opt.train_module)
    train_module = train_module.trainer(dp, opt)
except:
    pass    

if not hasattr(opt, 'critRecon'):
    opt.critRecon = 'BCELoss'
    
if not hasattr(opt, 'dtype'):
    opt.dtype = 'float'

# pdb.set_trace()
opt.gpu_ids = [0, 1]
models, optimizers, criterions, logger, opt = load_model(model_provider, opt)

enc = models['enc']
dec = models['dec']
enc.train(False)
dec.train(False)

for p in enc.parameters():
    p.requires_grad = False

for p in dec.parameters():
    p.requires_grad = False

models = None
optimizers = None


print('Done loading model.')

# Get the embeddings for the structure localization

opt.batch_size = 100
# opt.gpu_ids = [0,1,3]
enc.gpu_ids = opt.gpu_ids
dec.gpu_ids = opt.gpu_ids

embeddings_path = opt.save_dir + os.sep + 'embeddings_struct.pkl'
if os.path.exists(embeddings_path):
    embeddings = torch.load(embeddings_path)
else:
    embeddings = get_latent_embeddings(enc, dp, opt)
    torch.save(embeddings, embeddings_path)

# enc = None    
    
print('Done loading embeddings.')


#######    
### Main Loop
#######

import pdb
from aicsimage.io import omeTifWriter
from imgToProjection import imgtoprojection
from IPython.core.display import display
import PIL.Image
import matplotlib.pyplot as plt
import scipy.misc

import pandas as pd



opt.batch_size = 400
gpu_id = opt.gpu_ids[0]

loss = nn.MSELoss()
embeddings_all = torch.cat([embeddings['train'], embeddings['test']], 0);

dat_train_test = ['train'] * len(embeddings['train']) + ['test'] * len(embeddings['test'])
dat_dp_inds = np.concatenate([np.arange(0, len(embeddings['train'])), np.arange(0, len(embeddings['test']))], axis=0).astype('int')
dat_inds = np.concatenate([dp.data['train']['inds'], dp.data['test']['inds']])

colormap = 'hsv'
colors = plt.get_cmap(colormap)(np.linspace(0, 1, 4))


train_or_test_split = ['test', 'train']

img_paths_all = list()
err_save_paths = list()

test_mode = True

if test_mode:
    save_parent = opt.save_dir + os.sep + 'var_test_testing' + os.sep
    
    #do only 1000 samples
    nbatches = np.ceil(1000/opt.batch_size)
else: 
    save_parent = opt.save_dir + os.sep + 'var_test' + os.sep
    
if not os.path.exists(save_parent):
    os.makedirs(save_parent)


for train_or_test, i, img_index, c in tqdm(zip(dat_train_test, dat_dp_inds, dat_inds, range(0, len(dat_dp_inds))), 'computing errors', ascii=True):

    img_class = dp.image_classes[img_index]    
    img_class_onehot = dp.get_classes([i], train_or_test, 'onehot')
    
    img_name = dp.get_image_paths([i], train_or_test)[0]    
    img_name = os.path.basename(img_name)
    img_name = img_name[0:img_name.rfind('.')]
    
    save_dir = save_parent + os.sep + train_or_test + os.sep + img_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    err_save_path = save_dir + os.sep + img_name + '.csv'
    err_save_paths.append(err_save_path)
    
    if os.path.exists(err_save_path):
        continue

    # print(str(c) + os.sep + str(len(dat_dp_inds)))
    #Load the image
    img_in = dp.get_images([i], train_or_test)
    img_in = Variable(img_in.cuda(gpu_id), volatile=True)

    img_struct = torch.index_select(img_in, 1, torch.LongTensor([1]).cuda(gpu_id))
    
    img_recon = dec(enc(img_in))
    img_recon_struct = torch.index_select(img_recon, 1, torch.LongTensor([1]).cuda(gpu_id))
    
    img_recon = None
    
    shape_embedding = embeddings[train_or_test][i]
    
    #set the class label so it is correct
    img_class_onehot_log = (img_class_onehot - 1) * 25

    mse_orig = list()
    mse_recon = list()
    
    pearson_orig = list()
    pearson_recon = list()
   
    embedding_index = list()
    embedding_train_or_test = list()
    
    nembeddings = embeddings_all.size()[0]
    inds = list(range(0,nembeddings))
    
    np.random.shuffle(inds)
    
    data_iter = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)]        
    for j in range(0, len(data_iter)):
        
        if test_mode and (j > nbatches): continue
        
        batch_inds = data_iter[j]
        batch_size = len(data_iter[j])

        embedding_index.append([dat_dp_inds[ind] for ind in batch_inds])
        embedding_train_or_test.append([dat_train_test[ind] for ind in batch_inds])
        
        z = [None] * 3
        z[0] = Variable(img_class_onehot_log.repeat(batch_size, 1).float().cuda(gpu_id), volatile=True)
        z[1] = Variable(shape_embedding.repeat(batch_size,1).cuda(gpu_id), volatile=True)

        struct_embeddings = embeddings_all.index(torch.Tensor(batch_inds).long())
        z[2] = Variable(struct_embeddings.cuda(gpu_id), volatile=True)

        imgs_out = dec(z)

        imgs_out = torch.index_select(imgs_out, 1, torch.LongTensor([1]).cuda(gpu_id))

        for img in imgs_out:
            mse_orig.append(loss(img, img_struct).data[0])
            mse_recon.append(loss(img, img_recon_struct).data[0])
            
            pearson_orig.append(pearsonr(img.view(-1), img_struct.view(-1)).data.cpu().numpy()[0])
            pearson_recon.append(pearsonr(img.view(-1), img_recon_struct.view(-1)).data.cpu().numpy()[0])
            
        del imgs_out
    
    
    embedding_index = np.concatenate(embedding_index)
    embedding_train_or_test = np.concatenate(embedding_train_or_test)
    
    nembeddings_tmp = len(embedding_index)
    
    tot_inten = torch.sum(img_struct).data[0]
    tot_inten_recon = torch.sum(img_recon_struct).data[0]
    
    data = [np.repeat(img_index, nembeddings_tmp), 
            np.repeat(i, nembeddings_tmp), 
            embedding_index, 
            embedding_train_or_test, 
            np.repeat(img_class, nembeddings_tmp), 
            np.repeat(img_name, nembeddings_tmp), 
            np.repeat(train_or_test, nembeddings_tmp), 
            np.repeat(tot_inten, nembeddings_tmp), 
            np.repeat(tot_inten_recon, nembeddings_tmp), 
            mse_orig, 
            mse_recon, 
            pearson_orig, 
            pearson_recon]
    
    columns = ['img_index', 'data_provider_index', 'embedding_data_provider_index', 'embedding_train_or_test', 'label', 'path', 'train_or_test', 'tot_inten', 'tot_inten_recon', 'mse_orig', 'mse_recon', 'pearson_orig', 'pearson_recon']
    
    df = pd.DataFrame(np.array(data).T, columns=columns)
    df.to_csv(err_save_path)
        
print('Done computing errors.')


save_all_path = save_parent + os.sep + 'all_dat.csv'



if not os.path.exists(save_all_path):
    csv_list = list()

    for err_save_path in tqdm(err_save_paths, 'loading error files', ascii=True):
    
        if os.path.exists(err_save_path):
            csv_errors = pd.read_csv(err_save_path)
    #                 csv_errors['train_or_test'] = train_or_test
            csv_list.append(csv_errors)
        else:
            print('Missing ' +  err_save_path)

    # pdb.set_trace()
    errors_all = pd.concat(csv_list, axis=0)

    print('Writing to ' + save_all_path)        
    errors_all.to_csv(save_all_path)
else:
    
    print(save_all_path + ' exists. Loading...')
    errors_all = pd.read_csv(save_all_path)

    
# ulabels = np.unique(errors_all['label'])




# from matplotlib import pyplot as plt

# plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')

# errors = errors_all.filter(regex='recon_err')
# errors_mean = errors.median(axis=1)

# errors_mean[np.isnan(errors_mean)] = math.huge

# errors_mean = np.divide(errors_mean, errors_all['tot_inten'])

# min_bin = np.prctile(errors_mean, 2)
# max_bin = np.prctile(errors_mean, 98)

# bins = np.linspace(min_bin, max_bin, 250)

# c = 0

# for train_or_test in train_or_test_split:
#     c+=1
#     plt.subplot(len(train_or_test_split), 1, c)
    
#     train_inds = errors_all['train_or_test'] == train_or_test
    
#     for label in ulabels:
#         label_inds = errors_all['label'] == label
        
#         inds = np.logical_and(train_inds, label_inds)
        
#         legend_key = label
#         plt.hist(errors_mean[inds], bins, alpha=0.5, label=legend_key, normed=True)
        
    
# plt.legend(loc='upper right')
# plt.show()



# from data_providers.DataProvider3D import load_h5 
# from model_utils import tensor2img
# from IPython.core.display import display
# import PIL.Image

# def get_images(dp, paths):
#     dims = list(dp.imsize)
#     dims[0] = len(dp.opts['channelInds'])

#     dims.insert(0, len(paths))

#     images = torch.zeros(tuple(dims))

#     if dp.opts['dtype'] == 'half':
#         images = images.type(torch.HalfTensor)

#     c = 0
#     for h5_path in paths:
#         image = load_h5(h5_path)
#         image = torch.from_numpy(image)
#         images[c] = image.index_select(0, torch.LongTensor(dp.opts['channelInds'])).clone()
#         c += 1

#     # images *= 2
#     # images -= 1
#     return images



# for label in ulabels:
#     print(label)
#     label_inds = errors_all['label'] == label

#     imgs_flat = list()
# #         label_inds = errors_all['label'] == 'Alpha tubulin'
#     for train_or_test in train_or_test_split:
# #         print(train_or_test)
#         train_inds = errors_all['train_or_test'] == train_or_test
#         inds = np.where(np.logical_and(train_inds, label_inds))

#         inds_sorted = np.argsort(errors_mean[inds[0]])

#         errors_sub = errors_all.loc[inds[0][inds_sorted]]

#         im_paths = [dp.image_paths[i] for i in errors_sub.iloc[0:10]['img_index']]
#         img_out = get_images(dp, im_paths)
#         img_flat_low_err = tensor2img(img_out)
        
#         im_paths = [dp.image_paths[i] for i in errors_sub.iloc[-10:]['img_index']]
#         img_out = get_images(dp, im_paths)
#         img_flat_hi_err = tensor2img(img_out)
    
#         imsize = img_flat_low_err.shape
#         border = np.ones([imsize[0], 10, 3])
    
#         img_flat = np.concatenate([img_flat_low_err, border, img_flat_hi_err], axis=1)
#         imgs_flat.append(img_flat)
    
#     display(PIL.Image.fromarray(np.uint8(np.concatenate(imgs_flat)*255)))


#     import matplotlib.pyplot as plt
    
    


# classes = dp.get_classes(np.arange(0, dp.get_n_dat('train')), 'train').numpy()
# embeddings_tmp = embeddings['train'].numpy()

# uclasses = np.unique(classes)

# ndims = embeddings['train'].shape[1]
# nrows = ndims/2


# plt.figure(figsize=(20, 2*nrows))

# counter = 1
           
# for dim in np.arange(0, ndims, 2):

#     for uclass in uclasses:

        
#         class_inds = classes == uclass;


#         plt.subplot(nrows, len(uclasses), counter)
#         plt.scatter(embeddings_tmp[class_inds,dim], embeddings_tmp[class_inds,dim+1], c=classes[class_inds], s=0.1)
#         plt.axis('equal')
#         plt.axis([-4, 4, -4, 4])
        
        
#         if uclass == 0:
#             plt.xlabel('x ' + str(dim))
#             plt.ylabel('x ' + str(dim+1))

#         if dim == 0: plt.title(dp.label_names[uclass])

#         counter += 1

# plt.savefig('{0}/latent_space.png'.format(model_dir), bbox_inches='tight')