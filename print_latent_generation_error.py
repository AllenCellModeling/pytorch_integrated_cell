#######    
### This function prints off the most likely predicted 
### channels for each of the cells in our dataset
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

import model_utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pdb

from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu id')
parser.add_argument('--batch_size', type=int, default=400, help='batch_size')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model' 

opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))
print(opt)

opt.gpu_ids = args.gpu_ids

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

dp = model_utils.load_data_provider(opt.data_save_path, opt.imdir, opt.dataProvider)

#######    
### Load REFERENCE MODEL
#######

opt.channelInds = [0, 1, 2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = dp.get_n_classes()
opt.nRef = opt.nlatentdim

models, optimizers, _, _, opt = model_utils.load_model(opt.model_name, opt)

enc = models['enc']
dec = models['dec']

enc.train(False)
dec.train(False)

models = None
optimizers = None


print('Done loading model.')

# Get the embeddings for the structure localization
opt.batch_size = 100
embeddings_path = opt.save_dir + os.sep + 'embeddings_struct.pyt'
embeddings = model_utils.load_embeddings(embeddings_path, enc, dp, opt)

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

from corr_stats import pearsonr, corrcoef
import pytorch_ssim



opt.batch_size = args.batch_size
gpu_id = opt.gpu_ids[0]

MSEloss = nn.MSELoss()
BCEloss = nn.BCELoss()

ssim_loss = pytorch_ssim.SSIM(window_size = 5)



embeddings_all = torch.cat([embeddings['train'], embeddings['test']], 0);

dat_train_test = ['train'] * len(embeddings['train']) + ['test'] * len(embeddings['test'])
dat_dp_inds = np.concatenate([np.arange(0, len(embeddings['train'])), np.arange(0, len(embeddings['test']))], axis=0).astype('int')
dat_inds = np.concatenate([dp.data['train']['inds'], dp.data['test']['inds']])


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
    img_struct = img_struct
    
    img_recon = dec(enc(img_in))
    img_recon_struct = torch.index_select(img_recon, 1, torch.LongTensor([1]).cuda(gpu_id))
    img_recon_struct = img_recon_struct
    
    img_recon = None
    
    shape_embedding = embeddings[train_or_test][i]
    
    #set the class label so it is correct
    img_class_onehot_log = (img_class_onehot - 1) * 25

    mse_orig = list()
    mse_recon = list()
    
    bce_orig = list()
    bce_recon = list()
    
    pearson_orig = list()
    pearson_recon = list()
   
    corr_orig = list()
    corr_recon = list()

    ssim_orig = list()
    ssim_recon = list()
    
    embedding_index = list()
    embedding_train_or_test = list()
    
    nembeddings = embeddings_all.size()[0]
    inds = list(range(0,nembeddings))
    
    np.random.shuffle(inds)
    
    data_iter = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)]       
    np.random.shuffle(data_iter)
    
    for j in range(0, len(data_iter)):
        
        if test_mode and (j >= nbatches): continue
        
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

        # img_struct_cpu = np.squeeze(img_struct.data.cpu().numpy())
        # img_recon_struct_cpu = np.squeeze(img_recon_struct.data.cpu().numpy())   
        
        for img in imgs_out:
            
            img = img.unsqueeze(0)
            
            mse_orig.append(MSEloss(img, img_struct).data[0])
            mse_recon.append(MSEloss(img, img_recon_struct).data[0])
            
            bce_orig.append(BCEloss(img, img_struct).data[0])
            bce_recon.append(BCEloss(img, img_recon_struct).data[0])
            
            pearson_orig.append(pearsonr(img.view(-1), img_struct.view(-1)).data.cpu().numpy()[0])
            pearson_recon.append(pearsonr(img.view(-1), img_recon_struct.view(-1)).data.cpu().numpy()[0])
            
            corr_orig.append(corrcoef(torch.stack([img.view(-1), img_struct.view(-1)]))[0,1].data.cpu().numpy()[0])
            corr_recon.append(corrcoef(torch.stack([img.view(-1), img_recon_struct.view(-1)]))[0,1].data.cpu().numpy()[0])
            
            img_cpu = np.squeeze(img.data.cpu().numpy())
                
            ssim_orig.append(ssim_loss(img, img_struct).data[0])
            ssim_recon.append(ssim_loss(img, img_recon_struct).data[0])
            
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
            bce_orig,
            bce_recon,
            pearson_orig, 
            pearson_recon,
            corr_orig,
            corr_recon,
            ssim_orig,
            ssim_recon]
    
    columns = ['img_index', 'data_provider_index', 'embedding_data_provider_index', 'embedding_train_or_test', 'label', 'path', 'train_or_test', 'tot_inten', 'tot_inten_recon', 'mse_orig', 'mse_recon', 'bce_orig', 'bce_recon', 'pearson_orig', 'pearson_recon', 'corr_orig', 'corr_recon', 'ssim_orig', 'ssim_recon']
    
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