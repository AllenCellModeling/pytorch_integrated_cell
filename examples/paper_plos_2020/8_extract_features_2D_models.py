#!/usr/bin/env python
# coding: utf-8

# # 2D model feature extraction

# ## Display a summary of the 2D models

# In[3]:


import pandas as pd

pd.options.display.width = 200
pd.options.display.max_colwidth = None

# Read df_master, a summary of the 25 2D models used in trained 1_model_compare_2D.py
strCSVFullFilename_2DModels = '/allen/aics/modeling/caleb/data/df_master.csv'
print(strCSVFullFilename_2DModels)

dfCSV_2DModels = pd.read_csv(strCSVFullFilename_2DModels)
print(dfCSV_2DModels[['beta', 'intensity_norm', 'suffix', 'model_dir']].sort_values(['beta', 'intensity_norm'], ascending = [True, True]))


# ## Configuring the script parameters

# In[ ]:


# 5 cells with save_imgs (all seg methods) = 1 hr
# 20 cells with save_imgs (1 seg method) = 12 mins
# 100 cells, no save_imgs, 3 betas = 7 mins
# 100 cells, no save_imgs, 4 betas = 9 mins

feats_parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/results/feats_caleb/'

intNumCells = 20  # Set to < 0 to select the entire test set

# methods = {'gt_zero', otsu', 'local_gaussian', 'local_mean', 'local_median', 'li', 'mean', 'all'}
seg_method_real = 'gt_zero'    # Should be gt_zero for real cells
seg_method_gen = 'local_mean'  # Should be local_mean for generated cells

# Whether to mask the intensity images before feature extraction
# TODO: Test whether this makes a difference for the generated cells
mask_intensity_features_real = True
mask_intensity_features_gen = True

save_imgs = False       # Whether to save the binary masks
figsize_hist = (16, 2)  # Large = (30, 4), small = (16, 2)

gpu_ids = [5]  # A list of available GPUs

# A list of 2D models to use
model_dirs = [
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_298/', 
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_299/', 
    
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_312/', 
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_313/', 
    
    #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_329/', 
    
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_330/', 
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_331/', 
    
    # All generated cells look the same since beta is too high, not a good model to use
    #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_378/', 
    #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_379/', 
]

debug = False


# In[ ]:


# Detect whether the script is running in Jupyter or in
# batch mode
def fnIsBatchMode(argDebug = False):
    try:
        get_ipython
        
    except:
        blnBatchMode = True
        if (argDebug): print('Batch mode detected')
        
    else:
        blnBatchMode = False
        print('Interactive mode detected')
        
    return blnBatchMode


# In[ ]:


if (not fnIsBatchMode()):
    # Automatically reload modules before code execution
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import datetime
from pytz import timezone

strTimeZone = 'America/Los_Angeles'

# Returns the current datetime
def fnNow():
    return datetime.datetime.now(timezone(strTimeZone))

# Returns a timestamp (for use in filenames) in the format:
# YYYY/MM/DD-HH:MM:SS
#def fnGenTimestamp(argDatetime = fnNow()):  # TODO: For some reason this doesn't work
def fnGenTimestamp(argDatetime = None):
    if (argDatetime == None):
        argDatetime = fnNow()
    return argDatetime.strftime('%Y%m%d-%H%M%S')

# Returns a date and time in human-readable format
#def fnGetDatetime(argDatetime = fnNow()):  # TODO: For some reason this doesn't work
def fnGetDatetime(argDatetime = None):
    if (argDatetime == None):
        argDatetime = fnNow()
    return argDatetime.strftime('%m/%d/%Y %H:%M:%S')


# In[ ]:


import os

# Checks to see if a specified directory exists. If not, create a new one
def fnOSMakeDir(argPath):
    if (not os.path.exists(argPath)):
        print('The specified directory does not exist. Creating a new one: {}'.format(argPath))
        os.mkdir(argPath)
        
def fnSplitFullFilename(argFullFilename, argDebug = False):
    strPath, strFilename = os.path.split(argFullFilename)
    strBasename, strExt = os.path.splitext(strFilename)
    
    return strPath, strBasename, strExt

# Returns a list of full filenames recursively from a parent directory,
# sorted and filtered based on argFileExt
def fnGetFullFilenames(argParentPath, argFileExt, argDebug = False):
    lstFullFilenames = []
    
    # Loop recursively using argParentPath as the starting point
    for strRoot, lstDirs, lstFilenames in os.walk(argParentPath):
        if (argDebug):
            print('strRoot = {}'.format(strRoot))
            print('lstDirs = {}'.format(lstDirs))
            print('lstFilenames = {}'.format(lstFilenames))
            print()
            
        if (len(lstFilenames) > 0):
            lstFilenames.sort()
            
            # Loop through the list of filenames and reconstruct
            # with their full paths
            for strFilename in lstFilenames:
                # Process only files with extension argFileExt
                if strFilename.endswith(argFileExt):
                    # Append full path to each filename
                    strFullFilename = os.path.join(strRoot, strFilename)
                    lstFullFilenames.append(strFullFilename)
                    if (argDebug): print('  {}'.format(strFullFilename))
                    
    lstFullFilenames.sort()
    
    return lstFullFilenames


# In[ ]:


import glob
import pickle
import json
import os
import pickle

import torch
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from integrated_cell import model_utils, utils
from integrated_cell.metrics.embeddings_reference import get_latent_embeddings
from integrated_cell.models.bvae import kl_divergence


def dim_klds(mus, sigmas):
    kl_dims = list()
    for mu, sigma in zip(mus, sigmas):
        _, kl_dim, _ = kl_divergence(mu.unsqueeze(0), sigma.unsqueeze(0))
        
        kl_dims.append(kl_dim)
    
    return np.vstack(np.vstack(kl_dims))    
    
    
def get_embeddings_for_model(suffix, model_dir, parent_dir, save_path, use_current_results, mode = "validate"):

    if not os.path.exists(save_path):
        if use_current_results:
            return None
            
        networks, dp, args = utils.load_network_from_dir(model_dir, parent_dir, suffix = suffix)

        recon_loss = utils.load_losses(args)['crit_recon']

        enc = networks['enc']
        dec = networks['dec']

        enc.train(False)
        dec.train(False)

        embeddings = get_latent_embeddings(enc, dec, dp, modes=[mode], recon_loss = recon_loss, batch_size = 32)

        torch.save(embeddings, save_path)
    else:
        embeddings = torch.load(save_path)

    return embeddings


def embeddings2elbo(embeddings, alpha=0.5, mode = "validate"):

    recon_per_point = torch.mean(embeddings[mode]['ref']['recon'], 1)
    kld_per_point =  embeddings[mode]['ref']['kld']
    
    elbo_per_point = -2*((1-alpha)*recon_per_point + alpha*kld_per_point)
    
    return elbo_per_point, recon_per_point, kld_per_point


def get_embeddings_for_dir(model_dir, parent_dir, use_current_results=False, suffixes = None, mode = 'validate'):
    model_paths = np.array(natsorted(glob.glob('{}/ref_model/enc_*'.format(model_dir))))
    
    inds = np.linspace(0, len(model_paths)-1).astype('int')
    
    model_paths = model_paths[inds]
    
    if suffixes is None:
        suffixes = [model_path.split('/enc')[1].split('.pth')[0] for model_path in model_paths]
    
    results_dir = '{}/results'.format(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    embeddings_list = list()
    
    logger_file = '{0}/ref_model/logger_tmp.pkl'.format(model_dir)
    
    if not os.path.exists(logger_file):
        return
    
    with open( logger_file, "rb" ) as fp:
        logger = pickle.load(fp)

    args_file = "{}/args.json".format(model_dir)
    with open(args_file, "r") as f:
        args = json.load(f)
    
    model_summaries = list()
    
    for suffix in suffixes:
        
        model_summary_path = "{}/ref_model/embeddings_{}{}_summary.pth".format(model_dir, mode, suffix)
        
#         if os.path.exists(model_summary_path):
#             with open(model_summary_path, "rb") as f:
#                 model_summary = pickle.load(f)
#         else:
        embeddings_path = "{}/ref_model/embeddings_{}{}.pth".format(model_dir, mode, suffix)
    
        embeddings = get_embeddings_for_model(suffix, model_dir, parent_dir, embeddings_path, use_current_results, mode = mode)

        if embeddings is None: continue

        opt = json.load(open( '{0}/args.json'.format(model_dir), "rb" ))

        iteration = int(suffix[1:])-1
        iteration_index = np.where(np.array(logger.log['iter']) == iteration)[0]

        if len(iteration_index) == 0:
            continue


        embeddings['beta'] = opt['kwargs_model']['alpha']
        embeddings['elbo'], embeddings['recon'], embeddings['kld'] = embeddings2elbo(embeddings, embeddings['beta'], mode = mode)

        klds_per_dim = dim_klds(embeddings[mode]['ref']['mu'], embeddings[mode]['ref']['sigma'])

        model_summary = {"iteration": iteration,
                "epoch": np.array(logger.log['epoch'])[iteration_index],
                "elbo": np.mean(embeddings['elbo'].numpy()),
                "recons": np.mean(embeddings['recon'].numpy()),
                "klds": np.mean(embeddings['kld'].numpy()),
                "klds_per_dim": np.mean(klds_per_dim, 0),
                "model_dir": model_dir,
                "label": model_dir.split('/')[-2],
                "suffix": suffix,
                "args": args}

        with open(model_summary_path, "wb") as f:
            pickle.dump(model_summary, f)

        model_summaries.append(model_summary)
            
    return model_summaries


# In[ ]:


if (not fnIsBatchMode()):
    # Set plotting style
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


#gpu_ids = [5]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ID) for ID in gpu_ids])
if len(gpu_ids) == 1:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


parent_dir = "/allen/aics/modeling/gregj/results/integrated_cell/"

model_parent = '{}/test_cbvae_beta_ref'.format(parent_dir)

#model_dirs = glob.glob('/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_*/')

# model_dirs = [
#     '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_298/', 
#     '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_299/', 
    
#     '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_312/', 
#     '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_313/', 
    
#     #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_329/', 
    
#     '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_330/', 
#     '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_331/', 
    
#     # All generated cells look the same since beta is too high, not a good model to use
#     #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_378/', 
#     #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_379/', 
# ]

        
save_dir = '{}/results'.format(model_parent)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
results_dir = save_dir
    
    

datStart = fnNow()
print(f'Started on {fnGetDatetime(datStart)}')
print()

data_list = list()
for i, model_dir in enumerate(model_dirs):
    print(model_dir)
    
    # do model selection based on validation data
    model_summaries = get_embeddings_for_dir(model_dir, parent_dir, use_current_results = False, mode='validate')

    if model_summaries is None:
        continue
        
    # find the best model    
    elbo = np.array([model_summary['elbo'] for model_summary in model_summaries])
    suffix = [model_summary['suffix'] for model_summary in model_summaries] 
    
    if len(elbo) == 0:
        continue
    
    max_ind = np.argmax(elbo)
    best_elbo = elbo[max_ind]
    best_suffix = suffix[max_ind]
    
    best_ind = int(max_ind)
    
    # get results for test data
    model_summaries = get_embeddings_for_dir(model_dir, parent_dir, use_current_results = False, mode = "test", suffixes=[best_suffix])
    
    iteration = np.array([model_summary['iteration'] for model_summary in model_summaries])
    epoch = np.array([model_summary['epoch'] for model_summary in model_summaries])
    elbo = np.array([model_summary['elbo'] for model_summary in model_summaries])
    recons = np.array([model_summary['recons'] for model_summary in model_summaries])
    klds = np.array([model_summary['klds'] for model_summary in model_summaries])
    args = [model_summary['args'] for model_summary in model_summaries]
    suffix = [model_summary['suffix'] for model_summary in model_summaries]    
    klds_per_dim = np.hstack([model_summary['klds_per_dim'] for model_summary in model_summaries])
    
    beta = args[0]['kwargs_model']['alpha']
    
    label = model_dir.split('/')[-2]
    
    model_summary = {"iteration": iteration,
                    "epoch": epoch,
                    "elbo": elbo,
                    "recons": recons,
                    "klds": klds,
                    "klds_per_dim": klds_per_dim,
                    "model_dir": model_dir,
                    "label": label,
                    "suffix": suffix,
                    "args": args,
                    "best_elbo": best_elbo,
                    "beta": beta}
    

    data_list.append(model_summary)

datEnd = fnNow()
print()
print(f'Ended on {fnGetDatetime(datEnd)}')

datDuration = datEnd - datStart
print(f'datDuration = {datDuration}')
print()


# In[ ]:


print(f'Num. models = {len(data_list)}, {type(data_list[0])}')

lstBeta = []

for dctModel in data_list:
    print(f"suffix = {dctModel['suffix']}: beta = {dctModel['beta']}")
    lstBeta.append(dctModel['beta'])
    
print(f'Sorted betas = {np.sort(lstBeta)}')


# In[ ]:


# Reorganize the data structure into something slightly more manageable 

import tqdm 
import matplotlib

import torch

from skimage.external.tifffile import imsave

ks = list(range(1,11))

cuda = True

# dims = 2048
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
# model = InceptionV3([block_idx])
# if cuda:
#     model.cuda()

#inception score stuff
# inception_dir = '{}/results/inception/'.format(model_parent)

#Sample a generated and real images into their own class folders
modes = ['train','test','validate']

im_paths_real = {}
im_scores_real = {}
im_paths_gen = {}

class_list = list()
path_list = list()
mode_list = list()

_, dp, _ = utils.load_network_from_dir(data_list[0]['model_dir'], parent_dir)
dp.image_parent = '/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'

class_list = np.array(class_list)
path_list = np.array(path_list)
mode_list = np.array(mode_list)

class_list_gen = class_list[mode_list == 'validate']

im_paths_gen = {}
im_scores_gen = {}

#sample n_train images and stick them into directories
for i, data in enumerate(data_list):    

    model_ind = 0
    
    if len(data['suffix']) == 0:
        continue
        
    #Make sure we get the hightest-ELBO model
        
    suffix = data['suffix'][model_ind]
    model_dir = data['model_dir']
    model_short = data['model_dir'].split('/')[-2]

    im_paths_gen[i] = {}
    im_scores_gen[i] = {}
    
    im_scores_gen[i]['model_dir'] = data['model_dir']
    im_scores_gen[i]['label'] = data['label']
    im_scores_gen[i]['suffix'] = data['suffix'][model_ind]    
    im_scores_gen[i]['elbo'] = data['elbo'][model_ind]
    im_scores_gen[i]['recon'] = data['recons'][model_ind]
    im_scores_gen[i]['kld'] = data['klds'][model_ind]
    im_scores_gen[i]['klds_per_dim'] = data['klds_per_dim'][model_ind]    
    im_scores_gen[i]['epoch'] = data['epoch'][model_ind]
    im_scores_gen[i]['im_path'] = '{}/ref_model/progress_{}.png'.format(model_dir, int(data['elbo'][model_ind]))
    im_scores_gen[i]['args'] = data['args'][model_ind]
    im_scores_gen[i]['beta'] = data['beta']
    
  


# In[ ]:


print(f'Num. models = {len(im_scores_gen)}\n{im_scores_gen[0]}')


# In[ ]:


import pandas as pd

for i in im_scores_gen:
    #log specific model architechure choices
    
    color = 'k'
    
    if im_scores_gen[i]['args']['dataProvider'] == 'RefDataProvider':
        im_scores_gen[i]['intensity_norm'] = 0
    elif im_scores_gen[i]['args']['dataProvider'] == 'RescaledIntensityRefDataProvider':
        im_scores_gen[i]['intensity_norm'] = 1
    else:
        raise error
        
    if im_scores_gen[i]['args']['dataProvider'] == 'RescaledIntensityRefDataProvider':
        marker = 'p'
    else:
        marker = '^'
        
#     im_scores_gen[i]['beta'] = im_scores_gen[i]['args']['kwargs_model']['beta']
    im_scores_gen[i]['marker'] = marker
    im_scores_gen[i]['color'] = color



for i in im_scores_gen:
    beta = im_scores_gen[i]['beta']
    im_scores_gen[i]['model_arch_str'] = rf"$ \beta $ = {beta}"
    
df_master = pd.DataFrame.from_dict([im_scores_gen[i] for i in im_scores_gen])    
df_master = df_master.sort_values('beta')


# for s in df_master['model_arch_str']: print(s)


# In[ ]:


#for s in df_master['model_arch_str']: print(s)
#print(df_master)
df_master[['beta', 'intensity_norm', 'suffix', 'model_dir']].sort_values(['beta', 'intensity_norm'])
#df_master.to_csv('~/df_master.csv', index = False)


# ## Print out the best-on-validation set models for each $\beta$

# In[ ]:



best_model = 'asdfasdfasdf'

for i, data in enumerate(data_list):
    if data['model_dir'] == "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae/2019-07-19-09:27:15/":
        best_model = i
        break

best_model = np.argmax(best_elbo)

for data in data_list:
# data = data_list[best_model]


    ind = 0
    save_dir = data['model_dir']

    print("model_dir = '{}'".format(data['model_dir']))
    print("parent_dir = '{}'".format(parent_dir))
    print("suffix = '{}'".format(data['suffix'][ind]))


# ## Do feature calculation for some subset of models

# In[ ]:


import tqdm
import pickle
import pandas as pd
import scipy.stats


import importlib as imp
import integrated_cell.utils.features
imp.reload(integrated_cell.utils)
imp.reload(integrated_cell.utils.features)

from integrated_cell.utils.features import im2feats
from aicsfeature.extractor.common import get_shape_features, get_intensity_features, get_skeleton_features

from scipy import ndimage
from skimage import data as skdata
#from skimage.filters import threshold_otsu, threshold_local, try_all_threshold
from skimage import filters, morphology, measure, color
from PIL import Image
import matplotlib.pyplot as plt

def reshape_subplots(fig, ax):
    num_subplots = len(ax.flatten())
    
    gs = matplotlib.gridspec.GridSpec(1, num_subplots)

    for index in range(num_subplots):
        ax[index].set_position(gs[index].get_position(fig))
        
    return

def process_binmask(binmask):
    # Generate label for unconnected objects in the binary mask
    labels = measure.label(binmask)
    assert(labels.max() != 0)  # There should be at least 1 CC
    
    # Use regionprops to extract the properties of each object
    region_props = measure.regionprops(labels)
    
    # Sort the areas of all objecs and get the label of the
    # largest one
    areas = [(obj_prop['label'], obj_prop['area']) for obj_prop in region_props]
    sorted_areas = sorted(areas, key=lambda x:x[1], reverse=True)
    max_label = sorted_areas[0][0]  # Get the label of the largest object
    
    # Mask out all objects except for the largest one
    single_mask = np.where(labels == max_label, 1, 0)
    
    # Fill in any holes in the largest object
    filled_mask = ndimage.binary_fill_holes(single_mask)
    
    return filled_mask    
    
def get_binmask(im, method='local_mean', local_block_size=199, debug=False):
    # methods = {'gt_zero', otsu', 'local_gaussian', 'local_mean', 'local_median', 'li', 'mean', 'all'}
    
    if (method == 'gt_zero' or method == 'all'):
        # Estimate the optimal threshold
        flt_threshold_gt_zero = 0
        
        # Threshold the image to get the binary mask
        im_binmask_gt_zero = im > flt_threshold_gt_zero
        
        # Process the binary mask (e.g. keep the largest object, fill holes)
        im_binmask_gt_zero = process_binmask(im_binmask_gt_zero)
        im_binmask = im_binmask_gt_zero
        if (debug): print(f'segmentation method = {method}')
        
    if (method == 'otsu' or method == 'all'):
        # Estimate the optimal threshold
        flt_threshold_otsu = filters.threshold_otsu(im)
        
        # Threshold the image to get the binary mask
        im_binmask_otsu = im > flt_threshold_otsu
        
        # Process the binary mask (e.g. keep the largest object, fill holes)
        im_binmask_otsu = process_binmask(im_binmask_otsu)
        im_binmask = im_binmask_otsu
        if (debug): print(f'segmentation method = {method}')
        
    if (method == 'local_gaussian' or method == 'all'):
        flt_threshold_local_gaussian = filters.threshold_local(im, local_block_size, 'gaussian')
        im_binmask_local_gaussian = im > flt_threshold_local_gaussian
        im_binmask_local_gaussian = process_binmask(im_binmask_local_gaussian)
        im_binmask = im_binmask_local_gaussian
        if (debug): print(f'segmentation method = {method}')
        
    if (method == 'local_mean' or method == 'all'):
        flt_threshold_local_mean = filters.threshold_local(im, local_block_size, 'mean')
        im_binmask_local_mean = im > flt_threshold_local_mean
        im_binmask_local_mean = process_binmask(im_binmask_local_mean)
        im_binmask = im_binmask_local_mean
        if (debug): print(f'segmentation method = {method}')
        
    if (method == 'local_median' or method == 'all'):
        flt_threshold_local_median = filters.threshold_local(im, local_block_size, 'median')
        im_binmask_local_median = im > flt_threshold_local_median
        im_binmask_local_median = process_binmask(im_binmask_local_median)
        im_binmask = im_binmask_local_median
        if (debug): print(f'segmentation method = {method}')
        
    if (method == 'li' or method == 'all'):
        flt_threshold_li = filters.threshold_li(im)
        im_binmask_li = im > flt_threshold_li
        im_binmask_li = process_binmask(im_binmask_li)
        im_binmask = im_binmask_li
        if (debug): print(f'segmentation method = {method}')
        
    if (method == 'mean' or method == 'all'):
        flt_threshold_mean = filters.threshold_mean(im)
        im_binmask_mean = im > flt_threshold_mean
        im_binmask_mean = process_binmask(im_binmask_mean)
        im_binmask = im_binmask_mean
        if (debug): print(f'segmentation method = {method}')
        
    if (method == 'all'):
        return im_binmask_gt_zero, im_binmask_otsu, im_binmask_local_gaussian, im_binmask_local_mean, im_binmask_local_median, im_binmask_li, im_binmask_mean
    else:
        return im_binmask

def save_feats(im, save_path, seg_method='local_mean', mask_intensity_features=True, save_imgs=False, figsize_hist=(30, 4), figsize_cells=(30, 4), dpi=100, debug=False):
    
    assert im.shape[0] == 1 
    
    im_tmp = im[0].cpu().numpy()
    if debug:
        print(f'im_tmp.shape = {im_tmp.shape}')
        #imshow2ch(im_tmp)

    im_tmp = np.expand_dims(im_tmp, 3)
    if debug: print(f'im_tmp.shape = {im_tmp.shape}')

    im_struct = np.copy(im_tmp)
    if debug: print(f'im_struct.shape = {im_struct.shape}')
    
    for i in range(im_struct.shape[0]):
        if debug: print(f'np.max(im_struct[{i}]) = {np.max(im_struct[i])}')
        im_struct[i] = (im_struct[i] / np.max(im_struct[i]))*255
        
    im_struct = im_struct.astype('uint8')
    if debug: print(f'im_struct.shape = {im_struct.shape}')
    
    #if debug:
    #    imshow2ch(im_struct[:, :, :, 0]>0)
    #    imshow2ch(im_struct[:, :, :, 0])
        
    feats = {}
    
    if (seg_method == 'all'):
        cell_binmask_gt_zero, cell_binmask_otsu, cell_binmask_local_gaussian, cell_binmask_local_mean, cell_binmask_local_median, cell_binmask_li, cell_binmask_mean = get_binmask(im_struct[0, :, :, 0], method=seg_method, debug=debug)
        nuc_binmask_gt_zero, nuc_binmask_otsu, nuc_binmask_local_gaussian, nuc_binmask_local_mean, nuc_binmask_local_median, nuc_binmask_li, nuc_binmask_mean = get_binmask(im_struct[1, :, :, 0], method=seg_method, debug=debug)
        
        # Use local_mean method for feature extraction
        cell_binmask = cell_binmask_local_mean
        nuc_binmask = nuc_binmask_local_mean
        
    else:
        cell_binmask = get_binmask(im_struct[0, :, :, 0], method=seg_method, debug=debug)
        nuc_binmask = get_binmask(im_struct[1, :, :, 0], method=seg_method, debug=debug)
    
    #feats['dna_shape'] = get_shape_features(seg=im_struct[1]>0)
    feats['dna_shape'] = get_shape_features(seg=nuc_binmask[:, :, np.newaxis])
    if debug: print(f"dna_shape = {feats['dna_shape']}")
        
    if (mask_intensity_features):
        feats['dna_inten'] = get_intensity_features(img=np.where(nuc_binmask[:, :, np.newaxis] > 0, im_struct[1], 0))  # Apply binary mask to image
    else:
        feats['dna_inten'] = get_intensity_features(img=im_struct[1])
    if debug: print(f"dna_inten = {feats['dna_inten']}")
    
    #try:
    #    feats['dna_skeleton'] = get_skeleton_features(seg=im_struct[1])
    #except:
    #    feats['cell_skeleton'] = {}
    #if debug: print(f"dna_skeleton = {feats['dna_skeleton']}")

    #feats['cell_shape'] = get_shape_features(seg=im_struct[0]>0)
    feats['cell_shape'] = get_shape_features(seg=cell_binmask[:, :, np.newaxis])
    if debug: print(f"cell_shape = {feats['cell_shape']}")
        
    if (mask_intensity_features):
        feats['cell_inten'] = get_intensity_features(img=np.where(cell_binmask[:, :, np.newaxis] > 0, im_struct[0], 0))  # Apply binary mask to image
    else:
        feats['cell_inten'] = get_intensity_features(img=im_struct[0])
    if debug: print(f"cell_inten = {feats['cell_inten']}")
    
    #try:
    #    feats['cell_skeleton'] = get_skeleton_features(seg=im_struct[0])
    #except:
    #    feats['cell_skeleton'] = {}
    #if debug: print(f"cell_skeleton = {feats['cell_skeleton']}")
#     feats = im2feats(im_struct[0], im_struct[1], im_struct, extra_features=["io_intensity", "bright_spots", "intensity", "skeleton"])
    
    if (save_imgs):
        # Show and save histograms of input images (before and after normalization)
        objFig_hist, objAxes_hist = plt.subplots(1, 4, figsize=figsize_hist)

        objAxes_hist[0].hist(im_tmp[0, :, :, 0].flatten())
        objAxes_hist[0].set_yscale('log')
        objAxes_hist[0].set_title('cell_img')

        objAxes_hist[1].hist(im_tmp[1, :, :, 0].flatten())
        objAxes_hist[1].set_title('nuc_img')
        objAxes_hist[1].set_yscale('log')

        objAxes_hist[2].hist(im_struct[0, :, :, 0].flatten())
        objAxes_hist[2].set_yscale('log')
        objAxes_hist[2].set_title('cell_img_norm')

        objAxes_hist[3].hist(im_struct[1, :, :, 0].flatten())
        objAxes_hist[3].set_title('nuc_img_norm')
        objAxes_hist[3].set_yscale('log')

        plt.savefig(save_path.replace('.pkl', '_hist.png'), bbox_inches='tight', pad_inches=0, dpi=dpi)
        
        if (seg_method == 'all'):
            # Show and save input images (before and after segmentation)
            objFig, objAxes = plt.subplots(1, 16, figsize=figsize_cells)

            objAxes[0].imshow(im_tmp[0, :, :, 0], cmap='gray')
            objAxes[0].set_title('cell_img')

            objAxes[1].imshow(im_tmp[1, :, :, 0], cmap='gray')
            objAxes[1].set_title('nuc_img')

            #objAxes[2].imshow(im_struct[0, :, :, 0], cmap='gray')
            #objAxes[2].set_title('cell_img_norm')

            #objAxes[3].imshow(im_struct[1, :, :, 0], cmap='gray')
            #objAxes[3].set_title('nuc_img_norm')

            #objAxes[2].imshow(im_struct[0, :, :, 0] > 0, cmap='gray')
            objAxes[2].imshow(cell_binmask_gt_zero, cmap='gray')
            objAxes[2].set_title('> 0')

            #objAxes[3].imshow(im_struct[1, :, :, 0] > 0, cmap='gray')
            objAxes[3].imshow(nuc_binmask_gt_zero, cmap='gray')
            objAxes[3].set_title('> 0')

            objAxes[4].imshow(cell_binmask_otsu, cmap='gray')
            objAxes[4].set_title('Otsu')

            objAxes[5].imshow(nuc_binmask_otsu, cmap='gray')
            objAxes[5].set_title('Otsu')

            objAxes[6].imshow(cell_binmask_local_gaussian, cmap='gray')  # Good
            objAxes[6].set_title('Local (Gaussian)')

            objAxes[7].imshow(nuc_binmask_local_gaussian, cmap='gray')
            objAxes[7].set_title('Local (Gaussian)')

            objAxes[8].imshow(cell_binmask_local_mean, cmap='gray')  # Good
            objAxes[8].set_title('Local (Mean)')

            objAxes[9].imshow(nuc_binmask_local_mean, cmap='gray')
            objAxes[9].set_title('Local (Mean)')

            objAxes[10].imshow(cell_binmask_local_median, cmap='gray')
            objAxes[10].set_title('Local (Median)')

            objAxes[11].imshow(nuc_binmask_local_median, cmap='gray')
            objAxes[11].set_title('Local (Median)')

            objAxes[12].imshow(cell_binmask_li, cmap='gray')
            objAxes[12].set_title('Li')

            objAxes[13].imshow(nuc_binmask_li, cmap='gray')
            objAxes[13].set_title('Li')

            objAxes[14].imshow(cell_binmask_mean, cmap='gray')
            objAxes[14].set_title('Mean')

            objAxes[15].imshow(nuc_binmask_mean, cmap='gray')
            objAxes[15].set_title('Mean')
            
        else:
            # Show and save input images (before and after segmentation)
            objFig, objAxes = plt.subplots(1, 4, figsize=figsize_cells)

            objAxes[0].imshow(im_tmp[0, :, :, 0], cmap='gray')
            objAxes[0].set_title('cell_img')

            objAxes[1].imshow(im_tmp[1, :, :, 0], cmap='gray')
            objAxes[1].set_title('nuc_img')

            #objAxes[2].imshow(im_struct[0, :, :, 0], cmap='gray')
            #objAxes[2].set_title('cell_img_norm')

            #objAxes[3].imshow(im_struct[1, :, :, 0], cmap='gray')
            #objAxes[3].set_title('nuc_img_norm')

            #objAxes[2].imshow(im_struct[0, :, :, 0] > 0, cmap='gray')
            objAxes[2].imshow(cell_binmask, cmap='gray')
            objAxes[2].set_title(seg_method)

            #objAxes[3].imshow(im_struct[1, :, :, 0] > 0, cmap='gray')
            objAxes[3].imshow(nuc_binmask, cmap='gray')
            objAxes[3].set_title(seg_method)            

        for objAxis in objAxes:
            objAxis.axis('off')

        plt.savefig(save_path.replace('.pkl', '_img.png'), bbox_inches='tight', pad_inches=0, dpi=dpi)

        # NOTE: Li and mean thresholds give good results
        #fig, ax = filters.try_all_threshold(im_struct[0, :, :, 0], figsize=(20, 4), verbose=False)
        #reshape_subplots(fig, ax)
        #plt.savefig(save_path.replace('.pkl', '_cellseg.png'), bbox_inches='tight', pad_inches=0, dpi=dpi)

        #fig, ax = filters.try_all_threshold(im_struct[1, :, :, 0], figsize=(20, 4), verbose=False)
        #reshape_subplots(fig, ax)
        #plt.savefig(save_path.replace('.pkl', '_nucseg.png'), bbox_inches='tight', pad_inches=0, dpi=dpi)

        if (not fnIsBatchMode()):
            plt.show()

        plt.close('all')
    
    with open(save_path, "wb") as f:
        pickle.dump(feats, f)
    
    return

import re

def load_feats(save_paths):
    feats = list()
    for save_path in save_paths:
        with open(save_path, 'rb') as f:
            feat_tmp = pickle.load(f)
            
        feat = {}
        
        # Extract the cell_idx from the filename
        _, basename, _ = fnSplitFullFilename(save_path)
        result = re.match('feat_(\d+)', basename)
        cell_idx = result.groups()[0]
        feat['cell_idx'] = cell_idx
        
        for i in feat_tmp:
            for j in feat_tmp[i]:
                feat["{}_{}".format(i,j)] = feat_tmp[i][j]            
            
        feats.append(feat)

    feats = pd.DataFrame.from_dict(feats)
        
    return feats


# In[ ]:


import integrated_cell.metrics.embeddings_reference as get_embeddings_reference

def fnLoadUnsortedEmbeddings(argUnsortedEmbeddingsPath = './dctUnsortedEmbeddings.pth', argEncoder = None, argDecoder = None, argDataProvider = None, argArgs = None, argReGen = False):
    # See if the embedding file already exists. If so, load it
    if (os.path.exists(argUnsortedEmbeddingsPath) and not(argReGen)):
        print('Loading unsorted embeddings file from: {}'.format(argUnsortedEmbeddingsPath))
        embeddings_ref_untreated_ref_mu = torch.load(argUnsortedEmbeddingsPath)
        
    # If the embedding file does not exist, generate one
    else:
        print('Generating unsorted embeddings file at: {}'.format(argUnsortedEmbeddingsPath))
        
        ref_enc = argEncoder
        ref_dec = argDecoder
        dp_ref  = argDataProvider

        recon_loss = utils.load_losses(argArgs)['crit_recon']
        batch_size = dp_ref.batch_size
        
        ### get embeddings for all cells and save to dict
        embeddings_ref_untreated_val = get_embeddings_reference.get_latent_embeddings(
            ref_enc,
            ref_dec,
            dp_ref,
            recon_loss,
            batch_size=batch_size,
            modes=['validate'],
        )
        embeddings_ref_untreated_test = get_embeddings_reference.get_latent_embeddings(
            ref_enc,
            ref_dec,
            dp_ref,
            recon_loss,
            batch_size=batch_size,
            modes=['test'],
        )
        embeddings_ref_untreated_train = get_embeddings_reference.get_latent_embeddings(
            ref_enc,
            ref_dec,
            dp_ref,
            recon_loss,
            batch_size=batch_size,
            modes=['train'],
        )
        embeddings_ref_untreated_ref_mu = {
            "train": embeddings_ref_untreated_train["train"]["ref"]["mu"],
            "test": embeddings_ref_untreated_test["test"]["ref"]["mu"],
            "validate": embeddings_ref_untreated_val["validate"]["ref"]["mu"],
        }

        torch.save(embeddings_ref_untreated_ref_mu, argUnsortedEmbeddingsPath)
        
    return embeddings_ref_untreated_ref_mu


# In[ ]:


import re

strEmbeddingsParentPath = '/allen/aics/modeling/caleb/data/'

def imshow2ch(im):
    objFig, (objAx1, objAx2) = plt.subplots(1, 2, figsize=(8, 5))
    objAx1.imshow(im[0, :, :], cmap='gray')
    objAx2.imshow(im[1, :, :], cmap='gray')
    
def fnGenUnsortedEmbeddingsPath(argEmbeddingsParentPath, argModelDir, argRefSuffix, argNumPathLevels=2):
    strModelName = '_'.join(re.sub('\/\/+', '/', argModelDir.strip('/').replace(':', '-')).split('/')[-argNumPathLevels:])  # Replace multiple slashes with single slash and split the path
    return argEmbeddingsParentPath + 'dctUnsortedEmbeddings_2DModel_' + strModelName + argRefSuffix + '.pth'
    
def fnGenRandomZ(argEmbeddingsParentPath, argModelDir, argRefSuffix, argBatchSize=1, argMode='train'):
    strEmbeddingsFullFilename = fnGenUnsortedEmbeddingsPath(argEmbeddingsParentPath, argModelDir, argRefSuffix)
    dctUnsortedEmbeddings = fnLoadUnsortedEmbeddings(strEmbeddingsFullFilename)
    embeddings_unfiltered = dctUnsortedEmbeddings[argMode]
    print(f'embeddings_unfiltered.shape = {embeddings_unfiltered.shape}')
    
    arrEmbeddingsMean = torch.mean(embeddings_unfiltered, dim=0, keepdim=True)
    arrEmbeddingsStdev = torch.std(embeddings_unfiltered, dim=0, keepdim=True)
    #print(f'{arrEmbeddingsMean.shape}, {arrEmbeddingsStdev.shape}')
    
    arrRandomZ = torch.normal(mean=arrEmbeddingsMean.repeat(argBatchSize, 1), std=arrEmbeddingsStdev.repeat(argBatchSize, 1))
    
    return arrRandomZ


# In[ ]:


from tqdm import tqdm

# All = (30, 4), single = (10, 4)
figsize_cells_real = (30, 4) if seg_method_real == 'all' else (10, 4)
figsize_cells_gen = (30, 4) if seg_method_gen == 'all' else (10, 4)

datStart = fnNow()
print(f'Started on {fnGetDatetime(datStart)}')
print()

#feats_parent_dir = "{}/feats/".format(results_dir)
#feats_parent_dir = "{}/feats_caleb/".format(results_dir)
print(f'feats_parent_dir = {feats_parent_dir}')

all_feats_save_path = "{}/all_feats.pkl".format(feats_parent_dir)
print(f'all_feats_save_path = {all_feats_save_path}')

intensity_norms = np.unique(df_master['intensity_norm'])

feature_path_dict = {}
#there are 2 normalization methods
for intensity_norm in intensity_norms:
    print(f'  intensity_norm = {intensity_norm}')
    
    #get the dataframe for this normalization method
    df_norm = df_master[df_master['intensity_norm'] == intensity_norm]
    
    #get the parent directory for saving this normalization method    
    save_norm_parent = "{}/norm_{}".format(feats_parent_dir, intensity_norm)
    print(f'  save_norm_parent = {save_norm_parent}')
    if not os.path.exists(save_norm_parent):
        os.makedirs(save_norm_parent)
        
    save_norm_feats = "{}/feats_test".format(save_norm_parent)
    print(f'  save_norm_feats = {save_norm_feats}')
    if not os.path.exists(save_norm_feats):
        os.makedirs(save_norm_feats)
        
    #get a data provider for this normalization methods
    networks, dp, args = utils.load_network_from_dir(df_norm['model_dir'].iloc[0], parent_dir, suffix=df_norm['suffix'].iloc[0])
    
    enc = networks['enc'].cuda()
    
    x = dp.get_sample()
    print(f'  x.shape = {x.shape}')
    
    z_tmp = enc(x.cuda())[0]
    z_tmp = z_tmp[[0]]
    print(f'  z_tmp.shape = {z_tmp.shape}')
    
    n_latent = z_tmp.shape[1]
    
    if (intNumCells <= 0):
        intNumCells = dp.get_n_dat('test')
        
    #cell_idx = range(dp.get_n_dat('test'))
    cell_idx = range(intNumCells)
    
    #n_dat = dp.get_n_dat('test')
    n_dat = len(cell_idx)
    

    #save_real_feats_paths = ['{}/feat_{}.pkl'.format(save_norm_feats, i) for i in range(n_dat)]
    save_real_feats_paths = ['{}/feat_{}.pkl'.format(save_norm_feats, i) for i in cell_idx]
    
    # Loop through all the real images (test set) and save them
    for i, save_real_feat_path in tqdm(enumerate(save_real_feats_paths)):
        if not os.path.exists(save_real_feat_path):
            print(f'    i = {i}, cell_idx = {cell_idx[i]}, save_real_feat_path = {save_real_feat_path}')
            #im = dp.get_sample('test', [i])   
            im = dp.get_sample('test', [cell_idx[i]])
            save_feats(im, 
                       save_real_feat_path, 
                       seg_method=seg_method_real, 
                       mask_intensity_features=mask_intensity_features_real, 
                       save_imgs=save_imgs, 
                       figsize_hist=figsize_hist, 
                       figsize_cells=figsize_cells_real, 
                       debug=debug)

    feature_path_dict[intensity_norm] = {}
    feature_path_dict[intensity_norm]['real'] = load_feats(save_real_feats_paths)
    feature_path_dict[intensity_norm]['gen'] = {}
    
    # now loop through all the models under this normalization method, saving generated images and features
    for i in range(df_norm.shape[0]):
        print(f'    i_df_norm = {i}')
        
        # ***BUG?: I think we should be using df_norm below, and not df_master
        #save_feats_dir = '{}/{}'.format(save_norm_parent, df_master['label'].iloc[i])
        save_feats_dir = '{}/{}'.format(save_norm_parent, df_norm['label'].iloc[i])
        print(f'    save_feats_dir = {save_feats_dir}')
        
        if not os.path.exists(save_feats_dir):
            os.makedirs(save_feats_dir)
        
        #load the network
        network_loaded = False
        
        # ***BUG?: Will this line always be using the network
        #          loaded in the last loop except for the first
        #          loop???
        #dec = networks['dec'].cuda()
        
        strModelDir = df_norm['model_dir'].iloc[i]
        strRefSuffix = df_norm['suffix'].iloc[i]
        
        beta = df_norm['beta'].iloc[i]
        print(f'    beta = {beta}')
        
        gen_real_path = f'{save_feats_dir}/real'
        gen_kld_path = f'{save_feats_dir}/kld'
        gen_norm_path = f'{save_feats_dir}/norm'
        
        os.makedirs(gen_real_path, exist_ok=True)
        os.makedirs(gen_kld_path, exist_ok=True)
        os.makedirs(gen_norm_path, exist_ok=True)
        
        save_gen_feats_paths_real = ['{}/feat_{}.pkl'.format(gen_real_path, i) for i in range(n_dat)]
        save_gen_feats_paths_kld = ['{}/feat_{}.pkl'.format(gen_kld_path, i) for i in range(n_dat)]
        save_gen_feats_paths_norm = ['{}/feat_{}.pkl'.format(gen_norm_path, i) for i in range(n_dat)]
        
        save_gen_feats_paths_dct = {
            'real': save_gen_feats_paths_real, 
            'rnd_kld': save_gen_feats_paths_kld, 
            'rnd_norm': save_gen_feats_paths_norm, 
        }
        
        # TODO:
        #   - test embeddings should be loaded and available here, and then integrated with the feature tables
        #   - z_tmps generated should be saved in save_feats_dir
        #   - images should also be saved in save_feats here
        
        # (1) Real cell features
        # (2) Gen cell features (from real cells) + latent space embeddings
        # (3) Gen cell features (random from unit gaussian) + latent space embeddings
        # (4) Gen cell features (ramdom from latent space dims) + latent space embeddings

        #import ipdb; ipdb.set_trace()
        
        strEmbeddingsFullFilename = fnGenUnsortedEmbeddingsPath(strEmbeddingsParentPath, strModelDir, strRefSuffix)
        dctUnsortedEmbeddings = fnLoadUnsortedEmbeddings(strEmbeddingsFullFilename)
        z_test = dctUnsortedEmbeddings['test'][cell_idx, :]
        print(f'    z_test.shape = {z_test.shape}')
        
        z_rnd_kld = fnGenRandomZ(strEmbeddingsParentPath, strModelDir, strRefSuffix, argBatchSize=n_dat)
        print(f'    z_rnd_kld.shape = {z_rnd_kld.shape}')
        
        z_rnd_norm = z_tmp[0, :].repeat(n_dat, 1)
        z_rnd_norm.normal_()
        print(f'    z_rnd_norm.shape = {z_rnd_norm.shape}')
        
        # Save embeddings (test set or random) to embeddings_z_test/rnd_kld/rnd_norm.pkl
        with open(f'{gen_real_path}/embeddings.pkl', "wb") as f:
            pickle.dump(z_test, f)
        print(f'    Saving embeddings (real) to {gen_real_path}/embeddings.pkl')
        
        with open(f'{gen_kld_path}/embeddings.pkl', "wb") as f:
            pickle.dump(z_rnd_kld, f)
        print(f'    Saving embeddings (rnd_kld) to {gen_kld_path}/embeddings.pkl')
        
        with open(f'{gen_norm_path}/embeddings.pkl', "wb") as f:
            pickle.dump(z_rnd_norm, f)
        print(f'    Saving embeddings (rnd_norm) to {gen_norm_path}/embeddings.pkl')
        
        feature_path_dict[intensity_norm]['gen'][beta] = {}
        
        for key in save_gen_feats_paths_dct.keys():
            
            save_gen_feats_paths = save_gen_feats_paths_dct[key]
            
            if (key == 'real'):
                z_tmp = z_test
                
            elif (key == 'rnd_kld'):
                z_tmp = z_rnd_kld
                
            else:
                z_tmp = z_rnd_norm
            
            for j, save_path in tqdm(enumerate(save_gen_feats_paths)):

                if not os.path.exists(save_path):
                    print(f'      j = {j}, save_path = {save_path}')

                    if not network_loaded:
                        networks, dp, args = utils.load_network_from_dir(df_norm['model_dir'].iloc[i], parent_dir, suffix=df_norm['suffix'].iloc[i])
                        dec = networks['dec'].cuda()  # ***BUG?: How come we load the network without assigning the enc or dec even though we use it in the next few lines?
                        network_loaded = True

                    with torch.no_grad():
                        # Randomly sample from a normal distribution (default mean = 0, stdev = 1)
                        # Does this make sense given that the latent space is not necessarily a unit
                        # Gaussian distribution?
                        #im = dec(z_tmp.normal_())
                        im = dec(z_tmp[j, :][np.newaxis, :].cuda())
                        print(f'      z_tmp.shape = {z_tmp.shape}\nim.shape = {im.shape}')

                    save_feats(im, 
                               save_path, 
                               seg_method=seg_method_gen, 
                               mask_intensity_features=mask_intensity_features_gen, 
                               save_imgs=save_imgs, 
                               figsize_hist=figsize_hist, 
                               figsize_cells=figsize_cells_gen, 
                               debug=debug)

            #beta = df_norm['beta'].iloc[i]
            #print(f'    beta = {beta}')

            #feature_path_dict[intensity_norm]['gen'][beta][key] = load_feats(save_gen_feats_paths)
            
            feature_path_dict[intensity_norm]['gen'][beta][key] = {}
            feature_path_dict[intensity_norm]['gen'][beta][key]['embeddings'] = z_tmp.cpu().detach().numpy()
            feature_path_dict[intensity_norm]['gen'][beta][key]['features'] = load_feats(save_gen_feats_paths)
            
with open(all_feats_save_path, "wb") as f:
    pickle.dump(feature_path_dict, f)

datEnd = fnNow()
print()
print(f'Ended on {fnGetDatetime(datEnd)}')

datDuration = datEnd - datStart
print(f'datDuration = {datDuration}')
print()

