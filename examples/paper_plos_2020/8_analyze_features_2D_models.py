#!/usr/bin/env python
# coding: utf-8

# # 2D model feature analysis

# ## Configuring the script parameters

# In[ ]:


# Automatically reload modules before code execution
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import os
import pickle
import pandas as pd
from PIL import Image

import numpy as np

import features_lib as flib


# In[ ]:


import matplotlib.pyplot as plt

# Set plotting style
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ## Notes on intensity normalization

# ```
# # 2D PNGs in 8-bit (0->255) with individual distributions for each channel
# imdir =                 '/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'
# args.json:              'imdir': '/raid/shared/ipp/scp_19_04_10/'  # Only on isilon
# ref_model/args_dp.json: 'im_dir': '/raid/shared/ipp/scp_19_04_10/'  # Only on isilon
#         
# utils/utils.py/load_network_from_dir:    args['imdir'] = '/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'  # CC
# utils/utils.py/load_network_from_dir:    dp_kwargs['im_dir'] = '/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'  # CC
# 
# # In 1_model_compare_2d.ipynb
#     if im_scores_gen[i]['args']['dataProvider'] == 'RefDataProvider':
#         im_scores_gen[i]['intensity_norm'] = 0
#     elif im_scores_gen[i]['args']['dataProvider'] == 'RescaledIntensityRefDataProvider':
#         im_scores_gen[i]['intensity_norm'] = 1
#         
# # Congif files
# # dp arg normalize_intensity not specified in config files, so the default is used:
# data_providers/DataProvider.py/load_image:  im = im / 255
# args_dp.json: 'RefDataProvider' for norm_0
#               'RescaledIntensityRefDataProvider' for norm_1
#                 "channel_intensity_values": [                                                               
#                     618.0294,                                                                               
#                     618.0294,                                                                               
#                     618.0294                                                                                
#                 ],
# 
# # During training
# 
# # RefDataProvider.py -> no further normalization
# 
# # RescaledIntensityRefDataProvider.py
# class DataProvider(ParentDataProvider):
#     # Same as DataProvider but zeros out channels indicated by the variable 'masked_channels'
#     def __init__(
#         self, channel_intensity_values=[2596.9521, 2596.9521, 2596.9521], **kwargs
#     ):
# 
#         super().__init__(**kwargs)
# 
#         self.channel_intensity_values = channel_intensity_values
# 
#     def get_sample(self, train_or_test="train", inds=None):
#         # returns
#         # x         is b by c by y by x
#         # x_class   is b by c by #classes
#         # graph     is b by c by c - a random dag over channels
# 
#         x, classes, ref = super().get_sample(train_or_test=train_or_test, inds=inds)
# 
#         for i in range(x.shape[0]):
#             for j in range(x.shape[1]):
#                 if torch.sum(x[i, j]) > 0:
#                     x[i, j] = x[i, j] * (
#                         self.channel_intensity_values[j] / torch.sum(x[i, j])
#                     )
# 
#         return x, classes, ref
# 
# get_sample (RefDataProvider) -> get images (rescale/crop) -> load_image (normalize_intensity) -> get_sample (RescaledIntensityDataProvider)
# 
# # In 8_extract_features_2D_models.ipynb
# main:        im = dp.get_sample('test', [cell_idx[i]])
# save_feats:  im_struct[i] = (im_struct[i] / np.max(im_struct[i]))*255 -> create binary mask
# 
# # For both real and generated cells:
# #   In save_feats, contrast-stretch to 255 and analyze features
# 
# # ipdb trace:
# '/raid/shared/ipp/scp_19_04_10//plate_3500001946/15508_128895_reg_flat.png'
# 
# In load_image:
#     
# ipdb> im.shape
# (3, 168, 104)
# ipdb> np.amax(im[0].flatten())
# 255.0
# ipdb> np.amax(im[1].flatten())
# 255.0
# ipdb> np.amax(im[2].flatten())
# 255.0
# ipdb> np.amin(im[0].flatten())
# 0.0
# ipdb> np.amin(im[1].flatten())
# 0.0
# ipdb> np.amin(im[2].flatten())
# 0.0
# 
# After: im = im / 255
# 
# ipdb> np.amax(im[0].flatten())
# 1.0
# ipdb> np.amin(im[0].flatten())
# 0.0
# 
# utils/utils.py/load_drug_data_provider:  image_parent = '/allen/aics/modeling/gregj/results/ipp/scp_drug_pilot_fixed/',
# 
# norm_0: img = 0->1, before feats = 0->255
# norm_1: img = 0->n, before feats = 0->255
# ```

# ```
# Printed from RescaledIntensityRefDataProvider.py
# Image/Channel: min/max/mean/std * factor = min/max/mean/std
# 
# 0/0: 0.00/1.00/0.05/0.10 * 0.79 = 0.00/0.79/0.04/0.08
# 0/1: 0.00/1.00/0.02/0.07 * 1.96 = 0.00/1.96/0.04/0.13
# 0/2: 0.00/1.00/0.03/0.08 * 1.56 = 0.00/1.56/0.04/0.13
# 1/0: 0.00/1.00/0.02/0.06 * 1.88 = 0.00/1.88/0.04/0.12
# 1/1: 0.00/1.00/0.06/0.13 * 0.69 = 0.00/0.69/0.04/0.09
# 1/2: 0.00/1.00/0.04/0.12 * 1.14 = 0.00/1.14/0.04/0.14
# 2/0: 0.00/1.00/0.03/0.06 * 1.51 = 0.00/1.51/0.04/0.09
# 2/1: 0.00/1.00/0.00/0.01 * 9.14 = 0.00/9.14/0.04/0.13
# 2/2: 0.00/1.00/0.05/0.16 * 0.88 = 0.00/0.88/0.04/0.14
# 3/0: 0.00/1.00/0.03/0.08 * 1.51 = 0.00/1.51/0.04/0.12
# 3/1: 0.00/1.00/0.01/0.07 * 4.70 = 0.00/4.70/0.04/0.31
# 3/2: 0.00/1.00/0.02/0.10 * 1.82 = 0.00/1.82/0.04/0.18
# 4/0: 0.00/1.00/0.04/0.09 * 1.02 = 0.00/1.02/0.04/0.09
# 4/1: 0.00/1.00/0.03/0.10 * 1.30 = 0.00/1.30/0.04/0.14
# 4/2: 0.00/1.00/0.03/0.12 * 1.29 = 0.00/1.29/0.04/0.15
# 5/0: 0.00/1.00/0.04/0.09 * 1.11 = 0.00/1.11/0.04/0.10
# 5/1: 0.00/1.00/0.03/0.10 * 1.36 = 0.00/1.36/0.04/0.13
# 5/2: 0.00/1.00/0.03/0.12 * 1.26 = 0.00/1.26/0.04/0.15
# 6/0: 0.00/1.00/0.06/0.09 * 0.71 = 0.00/0.71/0.04/0.07
# 6/1: 0.00/1.00/0.06/0.14 * 0.69 = 0.00/0.69/0.04/0.09
# 6/2: 0.00/1.00/0.07/0.17 * 0.60 = 0.00/0.60/0.04/0.10
# 7/0: 0.00/1.00/0.09/0.12 * 0.46 = 0.00/0.46/0.04/0.06
# 7/1: 0.00/1.00/0.09/0.12 * 0.46 = 0.00/0.46/0.04/0.06
# 7/2: 0.00/1.00/0.06/0.16 * 0.68 = 0.00/0.68/0.04/0.11
# 8/0: 0.00/1.00/0.06/0.12 * 0.73 = 0.00/0.73/0.04/0.09
# 8/1: 0.00/1.00/0.00/0.02 * 9.24 = 0.00/9.24/0.04/0.17
# 8/2: 0.00/1.00/0.03/0.11 * 1.36 = 0.00/1.36/0.04/0.14
# 9/0: 0.00/1.00/0.03/0.07 * 1.21 = 0.00/1.21/0.04/0.08
# 9/1: 0.00/1.00/0.03/0.07 * 1.20 = 0.00/1.20/0.04/0.09
# 9/2: 0.00/1.00/0.05/0.15 * 0.83 = 0.00/0.83/0.04/0.12
# 10/0: 0.00/1.00/0.03/0.07 * 1.20 = 0.00/1.20/0.04/0.09
# 10/1: 0.00/1.00/0.04/0.14 * 1.07 = 0.00/1.07/0.04/0.15
# 10/2: 0.00/1.00/0.04/0.14 * 1.10 = 0.00/1.10/0.04/0.15
# 11/0: 0.00/1.00/0.02/0.06 * 1.72 = 0.00/1.72/0.04/0.11
# 11/1: 0.00/1.00/0.06/0.12 * 0.72 = 0.00/0.72/0.04/0.09
# 11/2: 0.00/1.00/0.04/0.13 * 0.98 = 0.00/0.98/0.04/0.13
# 12/0: 0.00/1.00/0.03/0.09 * 1.17 = 0.00/1.17/0.04/0.10
# 12/1: 0.00/1.00/0.01/0.07 * 4.39 = 0.00/4.39/0.04/0.29
# 12/2: 0.00/1.00/0.03/0.12 * 1.32 = 0.00/1.32/0.04/0.16
# 13/0: 0.00/1.00/0.03/0.07 * 1.16 = 0.00/1.16/0.04/0.09
# 13/1: 0.00/1.00/0.05/0.12 * 0.85 = 0.00/0.85/0.04/0.11
# 13/2: 0.00/1.00/0.06/0.17 * 0.72 = 0.00/0.72/0.04/0.12
# 14/0: 0.00/1.00/0.03/0.07 * 1.28 = 0.00/1.28/0.04/0.08
# 14/1: 0.00/1.00/0.03/0.12 * 1.37 = 0.00/1.37/0.04/0.17
# 14/2: 0.00/1.00/0.05/0.14 * 0.77 = 0.00/0.77/0.04/0.11
# 15/0: 0.00/1.00/0.05/0.09 * 0.83 = 0.00/0.83/0.04/0.08
# 15/1: 0.00/1.00/0.05/0.12 * 0.75 = 0.00/0.75/0.04/0.09
# 15/2: 0.00/1.00/0.06/0.14 * 0.73 = 0.00/0.73/0.04/0.10
# 16/0: 0.00/1.00/0.06/0.11 * 0.67 = 0.00/0.67/0.04/0.07
# 16/1: 0.00/1.00/0.09/0.16 * 0.43 = 0.00/0.43/0.04/0.07
# 16/2: 0.00/1.00/0.04/0.14 * 0.95 = 0.00/0.95/0.04/0.13
# 17/0: 0.00/1.00/0.02/0.06 * 2.11 = 0.00/2.11/0.04/0.12
# 17/1: 0.00/1.00/0.02/0.06 * 2.22 = 0.00/2.22/0.04/0.13
# 17/2: 0.00/1.00/0.02/0.08 * 1.95 = 0.00/1.95/0.04/0.16
# 18/0: 0.00/1.00/0.06/0.12 * 0.64 = 0.00/0.64/0.04/0.08
# 18/1: 0.00/1.00/0.03/0.11 * 1.53 = 0.00/1.53/0.04/0.17
# 18/2: 0.00/1.00/0.04/0.14 * 0.92 = 0.00/0.92/0.04/0.13
# 19/0: 0.00/1.00/0.05/0.14 * 0.75 = 0.00/0.75/0.04/0.10
# 19/1: 0.00/1.00/0.02/0.07 * 1.65 = 0.00/1.65/0.04/0.11
# 19/2: 0.00/1.00/0.02/0.12 * 1.78 = 0.00/1.78/0.04/0.22
# 20/0: 0.00/1.00/0.05/0.11 * 0.82 = 0.00/0.82/0.04/0.09
# 20/1: 0.00/1.00/0.02/0.06 * 1.70 = 0.00/1.70/0.04/0.09
# 20/2: 0.00/1.00/0.03/0.11 * 1.37 = 0.00/1.37/0.04/0.15
# 21/0: 0.00/1.00/0.02/0.06 * 1.82 = 0.00/1.82/0.04/0.11
# 21/1: 0.00/1.00/0.04/0.11 * 0.92 = 0.00/0.92/0.04/0.10
# 21/2: 0.00/1.00/0.02/0.10 * 2.40 = 0.00/2.40/0.04/0.24
# 22/0: 0.00/1.00/0.04/0.09 * 0.94 = 0.00/0.94/0.04/0.09
# 22/1: 0.00/1.00/0.03/0.06 * 1.30 = 0.00/1.30/0.04/0.08
# 22/2: 0.00/1.00/0.04/0.15 * 1.09 = 0.00/1.09/0.04/0.16
# 23/0: 0.00/1.00/0.04/0.09 * 0.90 = 0.00/0.90/0.04/0.08
# 23/2: 0.00/1.00/0.04/0.12 * 0.95 = 0.00/0.95/0.04/0.12
# 24/0: 0.00/1.00/0.01/0.04 * 2.89 = 0.00/2.89/0.04/0.13
# 24/1: 0.00/1.00/0.01/0.05 * 3.44 = 0.00/3.44/0.04/0.16
# 24/2: 0.00/1.00/0.03/0.12 * 1.25 = 0.00/1.25/0.04/0.14
# 25/0: 0.00/1.00/0.03/0.07 * 1.46 = 0.00/1.46/0.04/0.10
# 25/1: 0.00/1.00/0.03/0.12 * 1.19 = 0.00/1.19/0.04/0.15
# 25/2: 0.00/1.00/0.03/0.12 * 1.24 = 0.00/1.24/0.04/0.15
# 26/0: 0.00/1.00/0.04/0.09 * 0.92 = 0.00/0.92/0.04/0.09
# 26/1: 0.00/1.00/0.03/0.10 * 1.27 = 0.00/1.27/0.04/0.12
# 26/2: 0.00/1.00/0.03/0.11 * 1.35 = 0.00/1.35/0.04/0.14
# 27/0: 0.00/1.00/0.07/0.12 * 0.55 = 0.00/0.55/0.04/0.06
# 27/1: 0.00/1.00/0.01/0.03 * 3.35 = 0.00/3.35/0.04/0.10
# 27/2: 0.00/1.00/0.03/0.13 * 1.57 = 0.00/1.57/0.04/0.20
# 28/0: 0.00/1.00/0.05/0.11 * 0.77 = 0.00/0.77/0.04/0.08
# 28/1: 0.00/1.00/0.01/0.02 * 5.59 = 0.00/5.59/0.04/0.10
# 28/2: 0.00/1.00/0.03/0.11 * 1.22 = 0.00/1.22/0.04/0.14
# 29/0: 0.00/1.00/0.08/0.14 * 0.52 = 0.00/0.52/0.04/0.07
# 29/2: 0.00/1.00/0.04/0.13 * 1.09 = 0.00/1.09/0.04/0.14
# 30/0: 0.00/1.00/0.06/0.11 * 0.67 = 0.00/0.67/0.04/0.07
# 30/1: 0.00/1.00/0.02/0.06 * 1.85 = 0.00/1.85/0.04/0.11
# 30/2: 0.00/1.00/0.04/0.14 * 0.98 = 0.00/0.98/0.04/0.14
# 31/0: 0.00/1.00/0.02/0.07 * 1.64 = 0.00/1.64/0.04/0.11
# 31/1: 0.00/1.00/0.03/0.10 * 1.39 = 0.00/1.39/0.04/0.15
# 31/2: 0.00/1.00/0.04/0.14 * 1.13 = 0.00/1.13/0.04/0.16
# ```

# ## Display cell and binary mask images for real cells and different betas

# ### Configuring the appropriate parameters

# In[ ]:


#norm_intensity{0|1}/beta{real|float}/sampling{real|kld|norm}

# norm_intensity/real
# norm_intensity/gen/beta/real/[embeddings/features]
#                    beta/kld/[embeddings/features]
#                    beta/norm/[embeddings/features]

#feats_parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/results/feats_caleb_20cells_saveimgs_localmean/'
feats_parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/results/feats_caleb_20cells_saveimgs_localmean_seganalysis/'
#feats_parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell//test_cbvae_beta_ref/results/feats_caleb_5cells_saveimgs_allsegs/'

intensity_norm   = 0                              # Values = {0, 1}
show_betas       = ['real', 0.296, 0.663]  # Values = {'real', 0.010, 0.296, 0.663}
#show_betas       = ['real', 0.010, 0.296, 0.663]  # Values = {'real', 0.010, 0.296, 0.663}
#show_betas       = ['real']                      # Values = {'real', 0.010, 0.296, 0.663}
sampling         = 'real'                          # Values = {real, kld, norm}
img_type         = 'cell'                         # Value = {'hist', 'cell'}


# In[ ]:


# TODO: Image files are sorted by filename, and not by numerical order

file_suffix = flib.dct_img_type[img_type]

img_paths = []

for col_idx, col_beta in enumerate(show_betas):
    img_paths.append(flib.fnGetFullFilenames(flib.gen_feature_path(feats_parent_dir, intensity_norm, col_beta, sampling), file_suffix))
    
num_cells = len(img_paths[0])
num_betas = len(show_betas)

if (img_type == 'hist'):
    row_factor = 2
    col_factor = 10
else:
    if (num_betas == 1):
        col_factor = 60
        row_factor = 6
    else:
        col_factor = 10
        row_factor = 4

objFig, objAxes = plt.subplots(num_cells, num_betas, figsize=(col_factor * num_betas, row_factor * num_cells))

for row_idx in np.arange(num_cells):
    
    for col_idx in np.arange(num_betas):
        
        fn_img = img_paths[col_idx][row_idx]
        arrImg = Image.open(fn_img)
        
        if (num_betas == 1):
            objAxes[row_idx].imshow(arrImg, cmap='gray')
        else:
            objAxes[row_idx, col_idx].imshow(arrImg, cmap='gray')

    path, basename, ext = flib.fnSplitFullFilename(fn_img)
    if (num_betas == 1):
        objAxes[row_idx].set_ylabel(basename, fontsize=20)
    else:
        objAxes[row_idx, 0].set_ylabel(basename, fontsize=20)
            
for col_idx in np.arange(num_betas):
    if (show_betas[col_idx] == 'real'):
        title = f'real cells'
    else:
        title = f'gen cells (norm = {intensity_norm}, beta = {show_betas[col_idx]})'
        
    if (num_betas > 1):
        objAxes[0, col_idx].set_title(title, fontsize=20)

#for objAxis in objAxes.flatten():
#    objAxis.axis('off')

for objAxis in objAxes.flatten():
    objAxis.xaxis.set_ticks([])
    objAxis.yaxis.set_ticks([])
    objAxis.spines['top'].set_visible(False)
    objAxis.spines['bottom'].set_visible(False)
    objAxis.spines['left'].set_visible(False)
    objAxis.spines['right'].set_visible(False)


# ## Print latent space embeddings and cell/nucleus features as a dataframe

# In[ ]:


# Load the pre-processed and pre-saved cell and nucleus features

#all_feats_save_folder = '/allen/aics/modeling/gregj/results/integrated_cell//test_cbvae_beta_ref/results/feats_caleb_100cells/'
all_feats_save_folder = '/allen/aics/modeling/gregj/results/integrated_cell//test_cbvae_beta_ref/results/feats_caleb_1000cells/'
all_feats_fn = 'all_feats.pkl'

#all_feats_save_path = '/allen/aics/modeling/gregj/results/integrated_cell//test_cbvae_beta_ref/results/feats//all_feats.pkl'
all_feats_save_path = os.path.join(all_feats_save_folder, all_feats_fn)

intensity_norms = ['unnormalized', 'normalized']

with open(all_feats_save_path, "rb") as f:
    feature_dict = pickle.load(f)

for i, intensity_norm in enumerate(intensity_norms):

    #features for a specified intensity norm
    feature_dict[i]

    #each sub-dict has real and gen features
    real_feats = feature_dict[i]['real']

    feature_dict[i]['gen']

    #then gen feature dict has features for many betas
    betas = [k for k in feature_dict[i]['gen']]

#     print(betas[0])

    #features_for_the_first_beta = feature_dict[i]['gen'][betas[0]]
    features_for_the_first_beta = feature_dict[i]['gen'][betas[0]]['real']['features']

#     print(features_for_the_first_beta)

    #the real columns and the generated columns (for each beta) are the same
    for columnReal, columnGen in zip(real_feats.columns, features_for_the_first_beta):
        assert columnReal == columnGen


# ### Run this section to flatten the features dictionary and output the embeddings and features as dataframes. Otherwise, skip to the next section for the plots

# In[ ]:


# Flatten the nested features dictionary into a flat dictionary of 1 level

# Structure of the nested dictionary:
#   norm_intensity{0, 1}/real
#   norm_intensity{0, 1}/gen/beta/real/{embeddings, features}
#                            beta/kld/{embeddings, features}
#                            beta/norm/{embeddings, features}

dct_combined_df_flat = flib.flatten_dict(feature_dict)

# Display the keys in the flattened dictionary
for key in dct_combined_df_flat.keys():
    print(f'{key}')


# In[ ]:


# Select a single combination of parameters and genereate a combined embeddings
# and features dataframe

# TODO: Loop through all the keys in the flattened dictionary and save all the
#       dataframes into csv files, or just save the flattened dictionary to a
#       pickle file?

key = '1_gen_0.296_rnd_norm'
#key = '1_real'

#embeddings = feature_dict[0]['gen'][0.010]['real']['embeddings']
if (key + '_embeddings' in dct_combined_df_flat.keys()):
    embeddings = dct_combined_df_flat[key + '_embeddings']
    embeddings_shape = embeddings.shape

    embedding_colnames = ['z_' + str(dim) for dim in np.arange(embeddings_shape[1])]

    df_embeddings = pd.DataFrame(embeddings, columns=embedding_colnames)

    #df_features = feature_dict[0]['gen'][0.010]['real']['features']
    df_features = dct_combined_df_flat[key + '_features']

    df_combined = pd.concat([df_embeddings, df_features], axis=1)

else:
    # If we are looking at real cells from the test set, there will only be
    # features but no embeddings since these cells are not generated
    df_combined = dct_combined_df_flat[key]


# In[ ]:


# Visiualize the embeddings and features dataframe
df_combined


# In[ ]:


# Save the combined embeddings and features dataframe to a csv file
df_combined.to_csv(os.path.join(all_feats_save_folder, f'df_combined_{key}.csv'), index=False)


# In[ ]:


# List the shape of the embeddings arrays for all the different combinations
# of parameters for sanity check
for key in dct_combined_df_flat.keys():
    if ('_embeddings' in key):
        print(f'{key}: {dct_combined_df_flat[key].shape}')


# ## Print off real versus generated feature distributions

# In[ ]:


from sklearn import preprocessing
from scipy.stats import gaussian_kde
from matplotlib import cm
from matplotlib.lines import Line2D

# ***TODO: Iterate over different internsity_norms?
# ***BUG: norm = 0, beta = 0.99, col = cell_shape_volume, feats_real = small, feats_gen = 4947 for all cells
#         -> not a bug, all genereated cells (from real ones) look the same when beta is 0.99
intensity_norm_meth = [0, 1]

betas_to_plot = [0.010, 0.296, 0.663]
show_samplings = ['real', 'rnd_kld', 'rnd_norm']

cols_to_plot = [
    "cell_shape_volume",
    "cell_shape_surface_area",
    'cell_inten_intensity_mean',    
    "dna_shape_volume",
#     "dna_shape_surface_area",
    'dna_inten_intensity_std'


#     'cell_inten_intensity_median',
#     'cell_inten_intensity_std',
#     'dna_inten_intensity_mean',
#     'dna_inten_intensity_median',

]

n_betas_to_plot = len(betas_to_plot)
n_feats = len(cols_to_plot)
color_real = 'r'
colors_betas = cm.viridis(np.linspace(0, 1, n_betas_to_plot))

betas = np.array([k for k in feature_dict[0]['gen']])

#betas_to_plot = betas[np.linspace(0, len(betas)-1, n_betas_to_plot).astype(int)]
#print(f'betas = {betas}, beta_to_plot = {betas_to_plot}')

points_to_eval_kde = np.linspace(-3,10,100)

for intensity_norm in intensity_norm_meth:

    features = feature_dict[intensity_norm]

    print(f'norm_intensity = {intensity_norm}, betas = {betas_to_plot}')

    for sampling in show_samplings:
        #plt.figure(figsize=[10, 6])
        plt.figure(figsize=[20, 3])

        #for sampling_type in ['real', 'rnd_kld', 'rnd_norm']

        for i, col in enumerate(cols_to_plot):
            #plt.subplot(2, 3, i+1)
            plt.subplot(1, 6, i+1)
            plt.title(col.replace('_', ' ').replace('surface area', 'perimeter').replace('volume', 'area').replace(' inten ', ' ').replace('dna ', 'DNA '))

            feats_real = features['real'][col].values.reshape(-1,1)

            scalar = preprocessing.StandardScaler()
            scalar.fit(feats_real)

            feats_real = scalar.transform(feats_real).flatten()

            density_real = gaussian_kde(feats_real).evaluate(points_to_eval_kde)

            plt.plot(points_to_eval_kde, density_real, label='observed (z-scored)', color = color_real, zorder=1E10)
            plt.fill_between(points_to_eval_kde, density_real, alpha = 0.1, color = color_real, zorder=1E10)

            if i == 0:
                plt.ylabel('relative distribution')

            for j, beta_to_plot in enumerate(betas_to_plot):
                # ***TODO: Iterate over real, rnd_kld, and rnd_norm
                feats_gen = features['gen'][beta_to_plot][sampling]['features'][col].values.reshape(-1,1)

                feats_gen = scalar.transform(feats_gen).flatten()
                density = gaussian_kde(feats_gen).evaluate(points_to_eval_kde)

                plt.plot(points_to_eval_kde, density, label=rf'$ \beta $ = {beta_to_plot}', color = colors_betas[j])
                plt.fill_between(points_to_eval_kde, density, alpha = 0.1, color = colors_betas[j])


            plt.yticks([])

        #     if i == (len(cols_to_plot) - 1):
        #         plt.legend(bbox_to_anchor=(1.05, .95), frameon=False)

        #plt.subplot(2, 3, i+2)    
        plt.subplot(1, 6, i+2)    

        legend_elements = [Line2D([0], [0], color=color_real, lw=4, label='observed (z-scored)')] + [Line2D([0], [0], color=colors_betas[j], lw=4, label=rf'$ \beta $ = {beta_to_plot}') for j, beta_to_plot in  enumerate(betas_to_plot)]
        plt.gca().legend(handles=legend_elements, loc='center', frameon=False)

        plt.gca().text(0.5, 0.9, f'Sampling = {sampling}', fontsize=16, ha='center', va='center')

        plt.axis('off')

        plt.tight_layout()

        # TODO: Loop through all specified parameter combinations and save all figures into png files?
        #plt.savefig('{}/features_and_betas.png'.format(results_dir), bbox_inches='tight', dpi=90)
        plt.show()

    plt.close()

