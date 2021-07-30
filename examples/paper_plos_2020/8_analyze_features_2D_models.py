#!/usr/bin/env python
# coding: utf-8

# # 2D model feature analysis

# ## Configuring the script parameters

# In[ ]:


# Automatically reload modules before code execution
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


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


# ## Display cell and binary mask images for real cells and different betas

# ### Configuring the appropriate parameters

# In[ ]:


#norm_intensity{0|1}/beta{real|float}/sampling{real|kld|norm}

# norm_intensity/real
# norm_intensity/gen/beta/real/[embeddings/features]
#                    beta/kld/[embeddings/features]
#                    beta/norm/[embeddings/features]

feats_parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/results/feats_caleb_20cells_saveimgs_localmean/'
#feats_parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell//test_cbvae_beta_ref/results/feats_caleb_5cells_saveimgs_allsegs/'

intensity_norm   = 0                             # Values = {0, 1}
show_betas       = ['real', 0.010, 0.296, 0.663]  # Values = {'real', 0.010, 0.296, 0.663}
#show_betas       = ['real']  # Values = {'real', 0.010, 0.296, 0.663}
sampling         = 'kld'                        # Values = {real, kld, norm}
img_type         = 'cell'                        # Value = {'hist', 'cell'}


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

#all_feats_save_path = '/allen/aics/modeling/gregj/results/integrated_cell//test_cbvae_beta_ref/results/feats//all_feats.pkl'
all_feats_save_path = '/allen/aics/modeling/gregj/results/integrated_cell//test_cbvae_beta_ref/results/feats_caleb_100cells/all_feats.pkl'

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

# ***TODO: Iterate over different internsity_norms
# ***BUG: norm = 0, beta = 0.99, col = cell_shape_volume, feats_real = small, feats_gen = 4947 for all cells
#         -> not a bug, all genereated cells (from real ones) look the same when beta is 0.99
intensity_norm_meth = 1

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
print(f'betas = {betas}, beta_to_plot = {betas_to_plot}')

points_to_eval_kde = np.linspace(-3,10,100)

features = feature_dict[intensity_norm_meth]

print(f'norm_intensity = {intensity_norm_meth}, betas = {betas_to_plot}')

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

    #plt.savefig('{}/features_and_betas.png'.format(results_dir), bbox_inches='tight', dpi=90)
    plt.show()

plt.close()

