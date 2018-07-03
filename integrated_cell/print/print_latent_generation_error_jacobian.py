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
parser.add_argument('--gpu_ids', nargs='+', type=int, default=0, help='gpu id')
parser.add_argument('--batch_size', type=int, default=400, help='batch_size')
parser.add_argument('--use_current_results', type=bool, default=False, help='if true, dont compute errors, and construct master table')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model' 

walk_files = glob.glob(args.parent_dir + os.sep + 'analysis' + os.sep + 'walks' + os.sep + 'walk_*.pkl')

def get_jacobian(z, dec):
    
#     nfiles = len(walk_files)
    
#     walk_file_inds = np.random.choice(nfiles, npts)    
#     walk_row_inds = np.random.choice( len_of_walk-1, npts)
    
#     positions = np.zeros([npts*2, ndims])
    
#     for file_ind, row, index in zip(walk_file_inds, walk_row_inds, np.arange(0, npts)):
#         positions[[index*2, index*2+1]] = pickle.load(open(walk_files[file_ind], "rb" ))[[row,row+1]]

    start_pts = np.arange(0, len_of_walk, int(np.floor(len_of_walk/npts)))
    end_pts = start_pts + 1
    inds = np.concatenate([[i,j] for i,j in zip(start_pts, end_pts)],0)
    
    positions = pickle.load(open(walk_files[index], "rb" ))[inds]
        
    return positions
    



walk_shape =  pickle.load( open( walk_files[0], "rb" ) ).shape
len_of_walk = walk_shape[0]
ndims = walk_shape[1]


# get_compare_points(walk_files, 1000, len_of_walk, ndims, 
# pdb.set_trace()

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


opt.batch_size = args.batch_size
gpu_id = opt.gpu_ids[0]

MSEloss = nn.MSELoss()
BCEloss = nn.BCELoss()

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

save_parent = opt.save_dir + os.sep + 'var_test_walk' + os.sep

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
    
    save_dir = save_parent + os.sep + train_or_test + os.sep + img_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    err_save_path = save_dir + os.sep + img_name + '.csv'
    err_save_paths.append(err_save_path)
    
    if os.path.exists(err_save_path) or args.use_current_results:
        continue

    # print(str(c) + os.sep + str(len(dat_dp_inds)))
    #Load the image
    img_in = dp.get_images([i], train_or_test)
    img_in = Variable(img_in.cuda(gpu_id), volatile=True)

    z_orig = enc(img_in)
    img_recon = None
    
    #set the class label so it is correct
    img_class_onehot_log = (img_class_onehot - 1) * 50

    mse_orig, mse_recon = list(), list()
    bce_orig, bce_recon = list(), list()    
    pearson_orig, pearson_recon = list(), list()
    corr_orig, corr_recon = list(), list()
    embedding_index, embedding_train_or_test = list(), list()
                
        
    inds = np.arange(0, npts*2)
    data_iter = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)]       
    # np.random.shuffle(data_iter)
    
    walk_pts = get_compare_points(walk_files, npts, len_of_walk, ndims, c)
    
    for j in range(0, len(data_iter)):
        
        inds = data_iter[j]
        batch_size = len(inds)
        
        embeddings = walk_pts[inds]
        
        z = [None] * 3
        z[0] = Variable(img_class_onehot_log.repeat(batch_size, 1).float().cuda(gpu_id), volatile=True)
        z[1] = Variable(z_orig[1].data[0].repeat(batch_size,1).cuda(gpu_id), volatile=True)
        z[2] = Variable(torch.Tensor(embeddings).cuda(gpu_id), volatile=True)
        
        imgs_out = dec(z)

        imgs_out = imgs_out.index_select(1, Variable(torch.LongTensor([1]).cuda(gpu_id)))

        # img_struct_cpu = np.squeeze(img_struct.data.cpu().numpy())
        # img_recon_struct_cpu = np.squeeze(img_recon_struct.data.cpu().numpy())   
        
        for k in range(0, batch_size, 2):
            
            img = imgs_out[k].unsqueeze(0)
            img2 = imgs_out[k+1].unsqueeze(0)
            
            img = img.unsqueeze(0)
            
            mse_orig.append(MSEloss(img2, img).data[0])
            bce_orig.append(BCEloss(img2, img).data[0])
            pearson_orig.append(pearsonr(img2.view(-1), img.view(-1)).data.cpu().numpy()[0])
            
            corr_orig.append(corrcoef(torch.stack([img2.view(-1), img.view(-1)]))[0,1].data.cpu().numpy()[0])

        del imgs_out
    
    
    data = [np.repeat(img_index, npts), 
            np.repeat(i, npts), 
            np.repeat(img_class, npts), 
            np.repeat(img_name, npts), 
            np.repeat(train_or_test, npts), 
            mse_orig, 
            bce_orig,
            pearson_orig, 
            corr_orig]
    
    columns = ['img_index', 'data_provider_index', 'label', 'path', 'train_or_test', 'mse_orig', 'bce_orig', 'pearson_orig', 'corr_orig']
    
    df = pd.DataFrame(np.array(data).T, columns=columns)
    
    df.to_csv(err_save_path)
        
print('Done computing errors.')


save_all_path = save_parent + os.sep + 'all_dat.csv'
save_all_missing_path = save_parent + os.sep + 'all_dat_missing.csv'



# if not os.path.exists(save_all_path):
csv_list = list()
csv_missing_list = list()

for err_save_path in tqdm(err_save_paths, 'loading error files', ascii=True):

    if os.path.exists(err_save_path):
        csv_errors = pd.read_csv(err_save_path)
#                 csv_errors['train_or_test'] = train_or_test
        csv_list.append(csv_errors)
    else:
        # print('Missing ' +  err_save_path)
        csv_missing_list.append(err_save_path)

# pdb.set_trace()
errors_all = pd.concat(csv_list, axis=0)

print('Writing to ' + save_all_path)        
errors_all.to_csv(save_all_path)

csv_missing_list = pd.DataFrame(csv_missing_list)
csv_missing_list.to_csv(save_all_missing_path)
