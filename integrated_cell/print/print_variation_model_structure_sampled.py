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

import itertools


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1,2,3], help='gpu id')
parser.add_argument('--batch_size', type=int, default=500, help='batch_size')
parser.add_argument('--use_current_results', type=bool, default=False, help='if true, dont compute errors, and construct master table')
args = parser.parse_args()

model_ref_dir = args.parent_dir + os.sep + 'ref_model'
model_struct_dir = args.parent_dir + os.sep + 'struct_model' 

save_parent = args.parent_dir + os.sep + 'analysis' + os.sep + 'model_structure_variation_sampled'
if not os.path.exists(save_parent):
    os.makedirs(save_parent)

opt = pickle.load(open( '{0}/opt.pkl'.format(model_struct_dir), "rb" ))
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

embeddings_ref_path = model_struct_dir + os.sep + 'embeddings.pkl'
embeddings_ref = model_utils.load_embeddings(embeddings_ref_path)

embeddings_struct_path = model_struct_dir + os.sep + 'embeddings.pkl'
embeddings_struct = model_utils.load_embeddings(embeddings_struct_path)

opt.batch_size = args.batch_size
gpu_id = opt.gpu_ids[0]

ndat = dp.get_n_dat('train')
nlabels = dp.get_n_classes()

img_paths_all = list()
err_save_paths = list()
label_names_all = list()
  
class_list = np.arange(0, nlabels)
        
    
for label_id in class_list:

    label_name = dp.label_names[label_id]
    
    label_names_all.append(label_name)
    
    npts = 2000
#     npts = np.sum(dp.get_classes(np.arange(0, ndat), train_or_test).numpy() == label_id)
    
    
    err_save_path = save_parent + os.sep + 'var_' + label_name + '.pkl'
    err_save_paths.append(err_save_path)
    
    if args.use_current_results or os.path.exists(err_save_path):
        continue
        
    if not os.path.exists(save_parent):
        os.makedirs(save_parent)

    
    #set the class label so it is correct
    img_class_onehot_log = torch.Tensor(nlabels).fill_(-25)
    img_class_onehot_log[label_id] = 0
    
    inds_ref = np.arange(0, npts)
    iter_ref = [inds_ref[j:j+opt.batch_size] for j in range(0, len(inds_ref), opt.batch_size)]       
    
    inds_struct = np.random.choice(ndat, npts) 
    iter_struct = [inds_struct[j:j+opt.batch_size] for j in range(0, len(inds_struct), opt.batch_size)]       
    # np.random.shuffle(data_iter)

    imgs_out_tmp = list()
    
    for j in range(0, len(iter_ref)):
        
        inds_ref_tmp = iter_ref[j]
        ints_struct_tmp = iter_struct[j]
        
        batch_size = len(inds_ref_tmp)
        
        embeddings_ref_tmp = torch.Tensor(batch_size, embeddings_ref['train'].size()[1]).normal_()
        embeddings_struct_tmp = torch.Tensor(batch_size, embeddings_struct['train'].size()[1]).normal_()
        
        z = [None] * 3
        z[0] = Variable(img_class_onehot_log.repeat(batch_size, 1).float().cuda(gpu_id), volatile=True)
        z[1] = Variable(torch.Tensor(embeddings_ref_tmp).cuda(gpu_id), volatile=True)    
        z[2] = Variable(torch.Tensor(embeddings_struct_tmp).cuda(gpu_id), volatile=True)

        imgs_out = dec(z)
        imgs_out = imgs_out.index_select(1, Variable(torch.LongTensor([1]).cuda(gpu_id), volatile=True)).cpu()

        imgs_out_tmp.append(imgs_out)

    imgs_out_tmp = torch.cat(imgs_out_tmp, 0).cpu()
    corr_mat = corrcoef(imgs_out_tmp.view(int(npts), -1)).data.cpu().numpy()
    
    _, log_det = np.linalg.slogdet(corr_mat)
    log_det_scaled = log_det/npts
    
    data = {'label_id': label_id,
            'label_name': label_name,
            'train_or_test': 'sampled',
            'corr_mat': corr_mat,
            'log_det': log_det,
            'log_det_scaled': log_det_scaled}
    
    pickle.dump(data, open(err_save_path, 'wb'))
        
print('Done computing errors.')


save_info_path = save_parent + os.sep + 'info.csv'


info_list = np.concatenate([np.expand_dims(np.array(label_names_all),1),
                            np.expand_dims(np.array(class_list),1),  
                            np.expand_dims(np.array(err_save_paths),1)], axis=1)

info_list = pd.DataFrame(info_list, columns=['label_name', 'label_id', 'save_path'])
info_list.to_csv(save_info_path, index=False)


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
    

# from matplotlib import pyplot as plt
# import seaborn as sns

# errors = data_list['log_det']

# min_bin = np.percentile(errors, 1)
# max_bin = np.percentile(errors, 99)

# c = 0

# pdb.set_trace()

# for train_or_test in train_or_test_split:
#     c+=1
#     plt.subplot(len(train_or_test_split), 1, c)
    
#     train_inds = data_list['train_or_test'] == train_or_test
    
#     for label in ulabels:
#         label_inds = data_list['label'] == label
        
#         inds = np.logical_and(train_inds, label_inds)
        
#         legend_key = label
#         sns.kdeplot(errors[inds])
        
    
# plt.legend(loc='upper right')
# plt.savefig('{0}/distr.png'.format(save_parent), bbox_inches='tight')




    


