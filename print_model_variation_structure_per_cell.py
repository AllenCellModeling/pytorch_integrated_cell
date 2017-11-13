#######    
### This function prints off the most likely predicted 
### channels for each of the cells in our dataset
#######
import argparse

import importlib
import numpy as np

import os
import pickle
import glob

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils

import pdb
import pandas as pd
from corr_stats import pearsonr, corrcoef

#have to do this import to be able to use pyplot in the docker image
import time

import model_utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pdb

from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1,2,3], help='gpu id')
parser.add_argument('--batch_size', type=int, default=500, help='batch_size')
parser.add_argument('--use_current_results', type=bool, default=False, help='if true, dont compute errors, and construct master table')
parser.add_argument('--overwrite', type=bool, default=False, help='if true, overwrite')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model' 

save_parent = args.parent_dir + os.sep + 'analysis' + os.sep + 'model_structure_variation_per_cell'
if not os.path.exists(save_parent):
    os.makedirs(save_parent)

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

embeddings_path = model_dir + os.sep + 'embeddings.pkl'

embeddings = model_utils.load_embeddings(embeddings_path, enc, dp, opt)
embeddings = torch.cat([embeddings['train'], embeddings['test']])

opt.batch_size = args.batch_size
gpu_id = opt.gpu_ids[0]

ntrain = dp.get_n_dat('train')
ntest = dp.get_n_dat('test')
ndat = ntrain + ntest

dat_train_test = ['train'] * ntrain + ['test'] * ntest
dat_dp_inds = np.concatenate([np.arange(0, ntrain), np.arange(0, ntest)], axis=0).astype('int')
dat_inds = np.concatenate([dp.data['train']['inds'], dp.data['test']['inds']])


train_or_test_split = ['test', 'train']

img_paths_all = list()
err_save_paths = list()

test_mode = True

#do only 1000 samples
npts = 1000
nbatches = np.ceil(1000/opt.batch_size)
    
if not os.path.exists(save_parent):
    os.makedirs(save_parent)

dat_list = list(zip(dat_train_test, dat_dp_inds, dat_inds, range(0, len(dat_dp_inds))))
np.random.shuffle(dat_list)

for train_or_test, i, img_index, c in tqdm(dat_list, 'computing errors', ascii=True):

    img_class = dp.image_classes[img_index]    
    img_class_onehot = dp.get_classes([i], train_or_test, 'onehot')
    
    img_name = dp.get_image_paths([i], train_or_test)[0]    
    img_name = os.path.basename(img_name)
    img_name = img_name[0:img_name.rfind('.')]
    
    save_dir = save_parent + os.sep + train_or_test


    err_save_path = save_dir + os.sep + img_name + '.pkl'
    err_save_paths.append(err_save_path)
    
    if args.use_current_results or os.path.exists(err_save_path):
        continue
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # print(str(c) + os.sep + str(len(dat_dp_inds)))
    #Load the image
    img_in = dp.get_images([i], train_or_test)
    img_in = Variable(img_in.cuda(gpu_id), volatile=True)

    z_orig = enc(img_in)
    img_recon = None
    
    #set the class label so it is correct
    img_class_onehot_log = (img_class_onehot - 1) * 50

    inds = np.random.choice(ndat, npts) 
    data_iter = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)]       
    # np.random.shuffle(data_iter)

    imgs_out_tmp = list()
    
    for j in range(0, len(data_iter)):
        
        inds = data_iter[j]
        batch_size = len(inds)
        
        embeddings_short = embeddings[inds,:]
        
        z = [None] * 3
        z[0] = Variable(img_class_onehot_log.repeat(batch_size, 1).float().cuda(gpu_id), volatile=True)
        z[1] = Variable(z_orig[1].data[0].repeat(batch_size,1).cuda(gpu_id), volatile=True)    
        z[2] = Variable(torch.Tensor(embeddings_short).cuda(gpu_id), volatile=True)
        
        imgs_out = dec(z)
        imgs_out = imgs_out.index_select(1, Variable(torch.LongTensor([1]).cuda(gpu_id)))

        imgs_out_tmp.append(imgs_out)

    imgs_out_tmp = torch.cat(imgs_out_tmp, 0)
    corr_mat = corrcoef(imgs_out_tmp.view(npts, -1)).data.cpu().numpy()
    
    _, log_det = np.linalg.slogdet(corr_mat)
    log_det_scaled = log_det/npts
    
    data = {'img_index': img_index, 
            'data_provider_index': i, 
            'label': img_class, 
            'path': img_name,
            'train_or_test': train_or_test,
            'corr_mat': corr_mat,
            'log_det': log_det,
            'log_det_scaled': log_det_scaled}
    
    pickle.dump(data, open(err_save_path, 'wb'))
        
print('Done computing errors.')


save_all_path = save_parent + os.sep + 'all_dat.csv'
save_all_missing_path = save_parent + os.sep + 'all_dat_missing.csv'

# if not os.path.exists(save_all_path):
data_list = list()
data_missing_list = list()

if os.path.exists(save_all_path) & args.use_current_results:
    data_list = pd.read_csv(save_all_path)
else:
    for err_save_path in tqdm(err_save_paths, 'loading error files', ascii=True):

        if os.path.exists(err_save_path):
            try:
                data = pickle.load(open(err_save_path, 'rb'))
                data.pop('corr_mat', None)

                data_list.append(data)
            except:
                data_missing_list.append(err_save_path)
        else:
            # print('Missing ' +  err_save_path)
            data_missing_list.append(err_save_path)

    data_list = pd.DataFrame(data_list)

    print('Writing to ' + save_all_path)        
    data_list.to_csv(save_all_path)

    data_missing_list = pd.DataFrame(data_missing_list)
    data_missing_list.to_csv(save_all_missing_path)
    

from matplotlib import pyplot as plt
import seaborn as sns

errors = data_list['log_det']

min_bin = np.percentile(errors, 2)
max_bin = np.percentile(errors, 98)

c = 0

pdb.set_trace()

for train_or_test in train_or_test_split:
    c+=1
    plt.subplot(len(train_or_test_split), 1, c)
    
    train_inds = data_list['train_or_test'] == train_or_test
    
    for label in ulabels:
        label_inds = data_list['label'] == label
        
        inds = np.logical_and(train_inds, label_inds)
        
        legend_key = label
        sns.kdeplot(errors_mean[inds])
        
    
plt.legend(loc='upper right')
plt.savefig('{0}/distr.png'.format(save_parent), bbox_inches='tight')




    

