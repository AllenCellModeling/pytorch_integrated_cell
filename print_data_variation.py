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

save_dir = args.parent_dir + os.sep + 'analysis' + os.sep + 'data_variation'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))
print(opt)

opt.gpu_ids = args.gpu_ids

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

dp = model_utils.load_data_provider(opt.data_save_path, opt.imdir, opt.dataProvider)

pdb.set_trace()

label_names_all = list()
save_paths_all = list()

n_classes = dp.get_n_classes()

n_test = dp.get_n_dat('test')
n_train = dp.get_n_dat('train')

label_ids = list(range(0, n_classes))

for label_id in tqdm(label_ids, 'computing variation per class', ascii=True):

    label_name = dp.label_names[label_id]
    label_names_all.append(label_name)
    
    err_save_path = save_dir + os.sep + 'var_' + label_name + '.pkl'
    save_paths_all.append(err_save_path)
    
    if os.path.exists(err_save_path) and not args.overwrite:
        continue
    
    class_test_inds = np.where(dp.get_classes(np.arange(0, n_test), 'test') == label_id)[0]
    im_test = dp.get_images(class_test_inds, 'test')
    
    class_train_inds = np.where(dp.get_classes(np.arange(0, n_train), 'train') == label_id)[0]
    im_train = dp.get_images(class_train_inds, 'train')

    imgs = torch.cat([im_test, im_train],0)

    ndat = imgs.size(0)

    #find variation in structure
    imgs_struct = imgs.index_select(1, torch.LongTensor([1]))  
#     imgs_struct = Variable(imgs_struct, volatile=True) 
    imgs_struct = Variable(imgs_struct, volatile=True) 
   
    corr_mat_struct = corrcoef(imgs_struct.view(ndat, -1)).data.cpu().numpy()
    _, log_det_struct = np.linalg.slogdet(corr_mat_struct)
    log_det_scaled_struct = log_det_struct/ndat
    
    #find variation in reference structure
    imgs_ref = imgs.index_select(1, torch.LongTensor([0, 2]))  
#     imgs_ref = Variable(imgs_ref.cuda(args.gpu_ids[0]), volatile=True) 
    imgs_ref = Variable(imgs_ref, volatile=True) 
   
    corr_mat_ref = corrcoef(imgs_ref.view(ndat, -1)).data.cpu().numpy()
    _, log_det_ref = np.linalg.slogdet(corr_mat_ref)
    log_det_scaled_ref = log_det_ref/ndat
    
    
    data = {'label': label_id, 
            'label_name': label_name,
            'corr_mat_struct': corr_mat_struct,
            'log_det_struct': log_det_struct,
            'log_det_scaled_struct': log_det_scaled_struct,
            'corr_mat_ref': corr_mat_ref,
            'log_det_ref': log_det_ref,
            'log_det_scaled_ref': log_det_scaled_ref}
    
    pickle.dump(data, open(err_save_path, 'wb'))
        
print('Done computing errors.')

save_info_path = save_dir + os.sep + 'info.csv'
info_list = pd.DataFrame([[a,b,c] for a,b,c in zip(label_names_all, label_ids, save_paths_all)], columns=['label_name', 'label_id', 'save_path'])
info_list.to_csv(save_info_path, index=False)

# save_all_path = save_parent + os.sep + 'all_dat.csv'
# save_all_missing_path = save_parent + os.sep + 'all_dat_missing.csv'

# # if not os.path.exists(save_all_path):
# data_list = list()
# data_missing_list = list()

# if os.path.exists(save_all_path) & args.use_current_results:
#     data_list = pd.read_csv(save_all_path)
# else:
#     for err_save_path in tqdm(err_save_paths, 'loading error files', ascii=True):

#         if os.path.exists(err_save_path):
#             try:
#                 data = pickle.load(open(err_save_path, 'rb'))
#                 corr_mat = data['corr_mat']

#                 _, data['log_det'] = np.linalg.slogdet(corr_mat)

#                 data.pop('corr_mat', None)

#                 data_list.append(data)
#             except:
#                 data_missing_list.append(err_save_path)
#         else:
#             # print('Missing ' +  err_save_path)
#             data_missing_list.append(err_save_path)

#     data_list = pd.DataFrame(data_list)

#     print('Writing to ' + save_all_path)        
#     data_list.to_csv(save_all_path)

#     data_missing_list = pd.DataFrame(data_missing_list)
#     data_missing_list.to_csv(save_all_missing_path)
    

# from matplotlib import pyplot as plt
# import seaborn as sns

# errors = data_list['log_det']

# min_bin = np.percentile(errors, 2)
# max_bin = np.percentile(errors, 98)

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
#         sns.kdeplot(errors_mean[inds])
        
    
# plt.legend(loc='upper right')
# plt.savefig('{0}/distr.png'.format(save_parent), bbox_inches='tight')




    

