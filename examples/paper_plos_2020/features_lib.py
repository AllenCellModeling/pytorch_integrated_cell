#!/usr/bin/env python
# coding: utf-8

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
    
    # Contrast-stretch each channel independently to 8-bit (255)
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
    
    if (mask_intensity_features):
        cell_img = np.where(cell_binmask[:, :, np.newaxis] > 0, im_struct[0], 0)
        nuc_img = np.where(nuc_binmask[:, :, np.newaxis] > 0, im_struct[1], 0)
        
        cell_img_desc = 'cell_img_norm_seg'
        nuc_img_desc = 'nuc_img_norm_seg'
        
    else:
        cell_img = im_struct[0]
        nuc_img = im_struct[1]
        
        cell_img_desc = 'cell_img_norm'
        nuc_img_desc = 'nuc_img_norm'
    
    #feats['dna_shape'] = get_shape_features(seg=im_struct[1]>0)
    feats['dna_shape'] = get_shape_features(seg=nuc_binmask[:, :, np.newaxis])
    if debug: print(f"dna_shape = {feats['dna_shape']}")
        
    #if (mask_intensity_features):
    #    feats['dna_inten'] = get_intensity_features(img=np.where(nuc_binmask[:, :, np.newaxis] > 0, im_struct[1], 0))  # Apply binary mask to image
    #else:
    #    feats['dna_inten'] = get_intensity_features(img=im_struct[1])
    feats['dna_inten'] = get_intensity_features(img=nuc_img)
    if debug: print(f"dna_inten = {feats['dna_inten']}")
    
    #try:
    #    feats['dna_skeleton'] = get_skeleton_features(seg=im_struct[1])
    #except:
    #    feats['cell_skeleton'] = {}
    #if debug: print(f"dna_skeleton = {feats['dna_skeleton']}")

    #feats['cell_shape'] = get_shape_features(seg=im_struct[0]>0)
    feats['cell_shape'] = get_shape_features(seg=cell_binmask[:, :, np.newaxis])
    if debug: print(f"cell_shape = {feats['cell_shape']}")
        
    #if (mask_intensity_features):
    #    feats['cell_inten'] = get_intensity_features(img=np.where(cell_binmask[:, :, np.newaxis] > 0, im_struct[0], 0))  # Apply binary mask to image
    #else:
    #    feats['cell_inten'] = get_intensity_features(img=im_struct[0])
    feats['cell_inten'] = get_intensity_features(img=cell_img)
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

        #objAxes_hist[2].hist(im_struct[0, :, :, 0].flatten())
        objAxes_hist[2].hist(cell_img.flatten())
        objAxes_hist[2].set_yscale('log')
        objAxes_hist[2].set_title(cell_img_desc)
        # TODO: Set xlim? ax1.set_xlim([0, 5])

        #objAxes_hist[3].hist(im_struct[1, :, :, 0].flatten())
        objAxes_hist[3].hist(nuc_img.flatten())
        objAxes_hist[3].set_title(nuc_img_desc)
        objAxes_hist[3].set_yscale('log')

        plt.savefig(save_path.replace('.pkl', '_hist.png'), bbox_inches='tight', pad_inches=0, dpi=dpi)
        
        if (seg_method == 'all'):
            # Show and save input images (before and after segmentation)
            objFig, objAxes = plt.subplots(1, 16, figsize=figsize_cells)

            objAxes[0].imshow(im_tmp[0, :, :, 0], cmap='gray')
            objAxes[0].set_title('cell_img')

            objAxes[1].imshow(im_tmp[1, :, :, 0], cmap='gray')
            objAxes[1].set_title('nuc_img')

            #objAxes[2].imshow(cell_img, cmap='gray')
            #objAxes[2].set_title(cell_img_desc)

            #objAxes[3].imshow(nuc_img, cmap='gray')
            #objAxes[3].set_title(nuc_img_desc)

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


dct_img_type = {
    'hist': '_hist.png', 
    'cell': '_img.png', 
}

def gen_feature_path(feats_parent_dir, intensity_norm, beta, sampling):
    fn_norm = 'norm_' + str(intensity_norm)
    
    # If beta is 'real', there is no sampling
    if (beta == 'real'):
        fn_jobname = 'feats_test'
        fn_sampling = ''
    else:
        df_master = pd.read_csv('/allen/aics/modeling/caleb/data/df_master.csv')
        model_dir = df_master.query('beta == @beta & intensity_norm == @intensity_norm')['model_dir'].to_numpy()[0]
        
        fn_jobname = os.path.basename(os.path.normpath(model_dir))
        fn_sampling = sampling
        
    return os.path.join(feats_parent_dir, fn_norm, fn_jobname, fn_sampling)


# In[ ]:


def flatten_dict(in_dict, dict_out=None, parent_key=None, separator="_"):
    if dict_out is None:
        dict_out = {}

    for k, v in in_dict.items():
        k = str(k)
        #print(f'{k}')
        k = f"{parent_key}{separator}{k}" if parent_key else k
        #print(f'  {k}')
        if isinstance(v, dict):
            #print(f'  Recursing...')
            flatten_dict(in_dict=v, dict_out=dict_out, parent_key=k)
            continue

        #print(f'  dct = {k}')
        dict_out[k] = v

    return dict_out

