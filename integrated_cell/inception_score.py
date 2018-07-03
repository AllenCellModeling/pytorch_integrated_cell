#######    
### This function prints off the inception score
### for both the input images and generated images
#######

import argparse

import importlib
import numpy as np
import pandas as pd

import os
import pickle

import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#have to do this import to be able to use pyplot in the docker image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from IPython import display

import model_utils


import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pdb
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='gpu id')
parser.add_argument('--batch_size', type=int, default=200, help='gpu id')
parser.add_argument('--overwrite', type=bool, default=False, help='overwrite?')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model'
ref_dir = args.parent_dir + os.sep + 'ref_model'

save_dir = args.parent_dir + os.sep + 'analysis' + os.sep + 'inception_score' + os.sep
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
        

# logger_file = '{0}/logger_tmp.pkl'.format(model_dir)
opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))
print(opt)

opt.gpu_ids = args.gpu_ids
gpu_id = opt.gpu_ids[0]
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

models, _, _, _, opt = model_utils.load_model(opt.model_name, opt)

opt.batch_size = args.batch_size

enc = models['enc']
dec = models['dec']
enc.train(False)
dec.train(False)

models = None
optimizers = None

print('Done loading model.')

#######    
### Main Loop
#######

import pdb
from aicsimage.io import omeTifWriter
from imgToProjection import imgtoprojection
import PIL.Image
from aicsimage.io import omeTifWriter

import scipy.misc
import pandas as pd



##########
## data ##
##########
fname = os.path.join(save_dir,'im_class_log_probs_data.pickle')

if os.path.exists(fname) and not args.overwrite:
    pass
else:

    im_class_log_probs = {}

    # For train or test
    for train_or_test in ['test', 'train']:
        ndat = dp.get_n_dat(train_or_test)
        inds = np.arange(0, ndat)    

        pred_log_probs = np.zeros([ndat,opt.nClasses])

        iter_struct = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)] 

        # For each cell in the data split
    #     for i in tqdm(range(0, 1)):
        for i in tqdm(iter_struct, desc='data, ' + train_or_test):

            # Load the image
            img_in = dp.get_images(i, train_or_test)
            img_in = Variable(img_in.cuda(gpu_id), volatile=True)

            # pass forward through the model
            z = enc(img_in)
            
            p = z[0].data.cpu().numpy()
            pred_log_probs[i,:] = p

        im_class_log_probs[train_or_test] = pred_log_probs

    # save test and train preds

    with open(fname, 'wb') as handle:
        pickle.dump(im_class_log_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)


#############
# autoencoded #
#############

fname = os.path.join(save_dir,'im_class_log_probs_autoencode.pickle')

if os.path.exists(fname) and not args.overwrite:
    pass
else:

    im_class_log_probs = {}

    # For train or test
    for train_or_test in ['test', 'train']:
        ndat = dp.get_n_dat(train_or_test)
        inds = np.arange(0, ndat)      

        pred_log_probs = np.zeros([ndat,opt.nClasses])

        iter_struct = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)] 

        for i in tqdm(iter_struct, desc='autoencode, ' + train_or_test):

            # Load the image
            img_in = dp.get_images(i, train_or_test)
            img_in = Variable(img_in.cuda(gpu_id), volatile=True)

            # pass forward through the model
            z = enc(dec(enc(img_in)))

            p = z[0].data.cpu().numpy()
            pred_log_probs[i,:] = p

        im_class_log_probs[train_or_test] = pred_log_probs

    # save test and train preds

    with open(fname, 'wb') as handle:
        pickle.dump(im_class_log_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

#############
# generated #
#############
fname = os.path.join(save_dir,'im_class_log_probs_gen.pickle')



if os.path.exists(fname) and not args.overwrite:
    pass
else:

    im_class_log_probs = {}

    # For train or test
    for train_or_test in ['test', 'train']:
        ndat = dp.get_n_dat(train_or_test)
        inds = np.arange(0, ndat)      

        pred_log_probs = np.zeros([ndat,opt.nClasses])

        iter_struct = [inds[j:j+opt.batch_size] for j in range(0, len(inds), opt.batch_size)] 

        for i in tqdm(iter_struct, desc='gen, ' + train_or_test):

            npts = len(i)
            # Load the image
            class_ids = dp.get_classes(i, train_or_test)

            classes = Variable(torch.Tensor(npts, opt.nClasses).fill_(-25).cuda(gpu_id), volatile=True)
            
            for j, class_id  in zip(range(0, npts), class_ids):
                classes[j, class_id] = 0

            # sample random latent space vectors
            ref = Variable(torch.Tensor(npts, opt.nRef).normal_().cuda(gpu_id), volatile=True)
            struct = Variable(torch.Tensor(npts, opt.nRef).normal_().cuda(gpu_id), volatile=True)

            # generate a fake cell of corresponding class
            img_in = dec([classes, ref, struct])

            # pass forward through the model
            z = enc(img_in)
            p = z[0].data.cpu().numpy()
            pred_log_probs[i,:] = p

        im_class_log_probs[train_or_test] = pred_log_probs

    # save test and train preds

    with open(fname, 'wb') as handle:
        pickle.dump(im_class_log_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

#############
# print csv #
#############

dirname = args.parent_dir
fname = 'analysis/inception_score/im_class_log_probs_data.pickle'
with open(os.path.join(dirname, fname), 'rb') as handle:
    data_logprobs = pickle.load(handle)

fname = 'analysis/inception_score/im_class_log_probs_autoencode.pickle'
with open(os.path.join(dirname, fname), 'rb') as handle:
    autoencode_logprobs = pickle.load(handle)   

fname = 'analysis/inception_score/im_class_log_probs_gen.pickle'
with open(os.path.join(dirname, fname), 'rb') as handle:
    gen_logprobs = pickle.load(handle)  
    
    
    
    
def D_KL(P,Q):
    return -np.sum(P*np.log(Q/P))  

def inception_score_all_ims(P_yGx):
    p_y = np.mean(P_yGx, axis=0)
    KL_divs = np.array([D_KL(p_yGx,p_y) for p_yGx in P_yGx])
    return np.exp(np.mean(KL_divs))

def KL_divs_per_im(P_yGx):
    p_y = np.mean(P_yGx, axis=0)
    KL_divs = np.array([D_KL(p_yGx,p_y) for p_yGx in P_yGx])
    return KL_divs


df = pd.DataFrame(columns=['phase', 'inds_phase', 'inds_master', 
                           'structureProteinName',
                           'KLdiv_data', 'KLdiv_autoencode', 'KLdiv_gen'])

# Get all of the inception scores into a big list
data_list = list()
for phase in dp.data.keys():
    
    KL_divs_data = KL_divs_per_im(np.exp(data_logprobs[phase]))
    KL_divs_autoencode = KL_divs_per_im(np.exp(autoencode_logprobs[phase]))
    KL_divs_gen = KL_divs_per_im(np.exp(gen_logprobs[phase]))
    
    for ind_phase, ind_master in enumerate(tqdm(dp.data[phase]['inds'])):
        
        data = [phase, ind_phase, ind_master,
                              dp.label_names[dp.get_classes([ind_phase], phase)[0]],
                              KL_divs_data[ind_phase], KL_divs_autoencode[ind_phase], KL_divs_gen[ind_phase]]
        
        data_list.append(data)

df = pd.DataFrame(data_list, columns=['phase', 'inds_phase', 'inds_master', 
                   'structureProteinName',
                   'KLdiv_data', 'KLdiv_autoencode', 'KLdiv_gen'])  

# compute the inception scores for each class, and all classes for training, test, and generated data
inception_scores = list()

train_inds = df['phase'] == 'train';
test_inds = df['phase'] == 'test';

for label in dp.label_names:

    struct_inds = df['structureProteinName'] == label;
    
    all_train_inds = train_inds & struct_inds
    all_test_inds = test_inds & struct_inds

    incept_gen_train = np.exp(np.mean(df['KLdiv_gen'][all_train_inds]))
    incept_gen_test = np.exp(np.mean(df['KLdiv_gen'][all_test_inds]))


    incept_data_train = np.exp(np.mean(df['KLdiv_data'][all_train_inds]))
    incept_data_test = np.exp(np.mean(df['KLdiv_data'][all_test_inds]))

    incept_autoencode_train = np.exp(np.mean(df['KLdiv_autoencode'][all_train_inds]))
    incept_autoencode_test = np.exp(np.mean(df['KLdiv_autoencode'][all_test_inds]))     

    inception_scores.append([incept_data_train, incept_data_test, 
                             incept_autoencode_train, incept_autoencode_test, 
                             incept_gen_train, incept_gen_test])

    
incept_gen_train = np.exp(np.mean(df['KLdiv_gen'][train_inds]))
incept_gen_test = np.exp(np.mean(df['KLdiv_gen'][test_inds]))

incept_data_train = np.exp(np.mean(df['KLdiv_data'][train_inds]))
incept_data_test = np.exp(np.mean(df['KLdiv_data'][test_inds]))

incept_autoencode_train = np.exp(np.mean(df['KLdiv_autoencode'][train_inds]))
incept_autoencode_test = np.exp(np.mean(df['KLdiv_autoencode'][test_inds]))


inception_scores.append([incept_data_train, incept_data_test, incept_autoencode_train, incept_autoencode_test, incept_gen_train, incept_gen_test])
 
df_inception_scores = pd.DataFrame(inception_scores, index=list(dp.label_names) + ['all classes'], columns=['data train', 'data test', 'autoencoded data train', 'autoencoded data test', 'generated data train', 'generated data test'])


df_inception_scores.to_csv(save_dir + os.sep + 'inception_scores.csv')

df_inception_scores_sigfigs = df_inception_scores.round(decimals=3)
df_inception_scores_sigfigs.to_csv(save_dir + os.sep + 'inception_scores_sigfigs.csv')
