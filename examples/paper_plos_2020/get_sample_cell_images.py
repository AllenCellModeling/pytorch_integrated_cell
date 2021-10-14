#!/usr/bin/env python
# coding: utf-8

# ## Adapted from Jupyter notebook 7_latent_space_visualization.ipynb

# In[1]:


# Automatically reload modules before code execution
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Set plotting style
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[4]:


import os
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt

import tifffile

from integrated_cell.utils.plots import tensor2im, imshow
import integrated_cell
from integrated_cell import model_utils, utils


# In[5]:


pd.options.display.max_rows = 200


# In[88]:


def imshow_subplot(im, argAx, scale_channels=True, scale_global=True, color_transform=None):
    # assume CYX image
    im = tensor2im(im, scale_channels, scale_global)
    argAx.imshow(im)
    argAx.axis("off")


# In[6]:


gpu_ids = [7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ID) for ID in gpu_ids])
if len(gpu_ids) == 1:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

image_parent = '/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'

#load the reference model
parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/'
#parent_dir = '/allen/aics/modeling/caleb/results/integrated_cell/'

ref_model_dir = '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:27:04'
ref_suffix = '_94544'    
#ref_model_dir = '/allen/aics/modeling/rorydm/projects/pytorch_integrated_cell/examples/training_scripts/bvae3D_actk_ref_seg_mito_beta_1_2021-02-02'

networks, dp_ref, args_ref = utils.load_network_from_dir(ref_model_dir, parent_dir, suffix=ref_suffix)

ref_enc = networks['enc']
ref_dec = networks['dec']

#load the target model
target_model_dir = "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/"
target_suffix = '_93300'

networks, dp_target, args_target = utils.load_network_from_dir(target_model_dir, parent_dir, suffix=target_suffix)
    
target_enc = networks['enc']
target_dec = networks['dec']


results_dir = '{}/results/latent_space_vizualization/'.format(parent_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
print("Results dir: {}".format(results_dir))

save_dir = results_dir

recon_loss = utils.load_losses(args_target)['crit_recon']


# ## Get metadata for all splits

# In[7]:


# Add column ControlMask to indicate whether the dataframe row contains a control cell
dfControlMask = dp_ref.csv_data['save_reg_path'].str.contains('control')
dfControlMask.name = 'ControlMask'

dp_ref.csv_data = pd.concat([dp_ref.csv_data,dfControlMask], axis=1)


# In[8]:


dfs_split = {
    split: pd.concat(
        [
            dp_ref.csv_data.loc[
                dp_ref.data[split]["inds"],
                ["CellId", "PlateId", "CellIndex", 'Gene', 'Protein', 'ProteinDisplayName', 'Structure', 'StructureDisplayName', 'StructureShortName', 'ProteinId/Name', 'StructureId/Name', 'save_reg_path', 'ControlMask']  # CC: Add column for image path here?
            ].reset_index(
            ).rename(
                columns={"index":"UnsplitCsvIndex"}
            ),
            pd.DataFrame({"split":[split]*len(dp_ref.data[split]["inds"])})
        ],
        axis=1
    ) for split in dp_ref.data.keys()
}


# ## Get features for all splits

# In[9]:


import json

with open('/raid/shared/ipp/scp_19_04_10/feats_out_with_units.json') as f:
    feats = json.load(f)

df_feats_all = pd.DataFrame.from_dict(feats["rows_with_units"])
my_feats = ["CellId", "dna_volume", "cell_volume"]
df_feats = df_feats_all[my_feats].copy()

df_feats["cell_height"] = df_feats_all["cell_position_highest_z"] - df_feats_all["cell_position_lowest_z"]
df_feats["dna_height"] = df_feats_all["dna_position_highest_z"] - df_feats_all["dna_position_lowest_z"]


# ## Get mito annotations for each split

# In[10]:


dfs_mito_all = {
    split: pd.DataFrame(
        {
            feat:dp_ref.data[split][feat] for feat in ["inds", "CellId", "mito_state_resolved"]
        }
    ).rename(
        columns={"inds":"UnsplitCsvIndex"}
    ) for split in dp_ref.data.keys()
}


# ## Find the embeddings for each split

# In[11]:


ref_embeds_test_path = '/allen/aics/modeling/caleb/data/7_latent_space_visualization_ref_embeds_test.pth'

if (os.path.exists(ref_embeds_test_path)):
    embeds_test = torch.load(ref_embeds_test_path)
    
else:
    from integrated_cell.metrics.embeddings_reference import get_latent_embeddings

    embeds_test = get_latent_embeddings(
        ref_enc,
        ref_dec,
        dp_ref,
        recon_loss,
        modes=['test', 'validate', 'train'],
        batch_size=32,
    )
    
    torch.save(embeds_test, ref_embeds_test_path)


# In[12]:


dfs_embeds = {
    split: pd.DataFrame(
        embeds_test[split]["ref"]["mu"].numpy(),
        columns=[f"mu_{i}" for i in range(embeds_test[split]["ref"]["mu"].numpy().shape[1])]
    ) for split in embeds_test.keys()
}


# In[13]:


for split, df_embed in dfs_embeds.items():
    assert len(dfs_embeds[split]) == len(dfs_split[split])
    dfs_embeds[split]["UnsplitCsvIndex"] = dfs_split[split]["UnsplitCsvIndex"]


# In[14]:


print(f"CellIds (dp_ref) = {len(dp_ref.csv_data['CellId'])}")
print(f"Unique CellIds (dp_ref) = {len(dp_ref.csv_data['CellId'].unique())}")
print(f'Non-control cells (dp_ref) = {np.sum(~dfControlMask)}')
print(f"CellIds (df_feats_all) = {len(df_feats_all['CellId'])}")
print(f"Unique CellIds (df_feats_all) = {len(df_feats_all['CellId'].unique())}")


# ## Merge embeddings in to metadata

# In[15]:


# This is for training set only
df_embeddings_plus_meta_train = dfs_split["train"].merge(
    dfs_mito_all["train"]
).merge(
    dfs_embeds['train']
)


# In[16]:


# This is for training set only
df_embeddings_plus_meta_and_feats_train = dfs_split["train"].merge(
    dfs_mito_all["train"]
).merge(
    dfs_embeds['train']
).merge(
    df_feats, how='left'  # Some cells don't have features, but we still want to include those rows
)


# In[17]:


df_embeddings_plus_meta_train_pretty_names = df_embeddings_plus_meta_and_feats_train.rename(
    columns={
        "mito_state_resolved": "Mitotic state",
        'dna_volume': 'DNA volume',
        'cell_volume': 'Cell volume',
        'dna_height': 'DNA height',
        'cell_height': 'Cell height'
    }
)


# In[18]:


df_embeddings_plus_meta_train_pretty_names[
    df_embeddings_plus_meta_train_pretty_names["Mitotic state"] != 'u'
].sort_values(by="Mitotic state")


# ## Filter by mitotic state and structure (leaving out control cells)

# In[19]:


print(f"Mitotic states = {df_embeddings_plus_meta_train_pretty_names['Mitotic state'].unique()}")
print(f"Structures = {df_embeddings_plus_meta_train_pretty_names['Structure'].unique()}")


# In[282]:


mitotic_state = 'M0'
#mitotic_state = 'M6/M7'
#structure = 'Actin filaments'
structure = 'Microtubules'
#structure = 'Golgi'

df_train_filtered = df_embeddings_plus_meta_train_pretty_names.query("`Mitotic state` == @mitotic_state & Structure == @structure")
df_train_filtered_nocontrols = df_train_filtered[df_train_filtered['ControlMask'] == False]

df_train_filtered_nocontrols[['UnsplitCsvIndex', 'CellId', 'Mitotic state', 'Structure', 'save_reg_path']]


# ## Visualize cell images

# In[288]:


print(f'image_parent = {dp_ref.image_parent}, {dp_target.image_parent}')

#ind = df_train_filtered_nocontrols.index[5]
ind = 4297

ref_x = dp_ref.get_sample('train', [ind])
target_x = dp_target.get_sample('train', [ind])

#print(f"{df_train_filtered_nocontrols.loc[ind][['UnsplitCsvIndex', 'CellId', 'CellIndex', 'Mitotic state', 'Gene', 'Protein', 'ProteinDisplayName', 'Structure', 'StructureDisplayName', 'StructureShortName', 'ProteinId/Name', 'StructureId/Name', 'save_reg_path', 'ControlMask']]}")
print(f"{ind}: {df_train_filtered_nocontrols.loc[ind]['save_reg_path']}")

#print(f'ref_x.shape = {ref_x.shape}')
#print(f'target_x.shape = {target_x.shape}')

plt.figure(figsize=(6, 8))
imshow(ref_x, scale_channels=True, scale_global=False)
plt.figure(figsize=(6, 8))
imshow(target_x[0], scale_channels=True, scale_global=False)

plt.tight_layout()


# In[284]:


iterIdx = iter(df_train_filtered_nocontrols.index)
print(f'{len(df_train_filtered_nocontrols.index)}')


# In[287]:


print(f'image_parent = {dp_ref.image_parent}, {dp_target.image_parent}')

for intIdx in np.arange(10):
    #ind = df_train_filtered_nocontrols.index[20]
    ind = next(iterIdx)

    ref_x = dp_ref.get_sample('train', [ind])
    target_x = dp_target.get_sample('train', [ind])

    #print(f"{df_train_filtered_nocontrols.loc[ind][['UnsplitCsvIndex', 'CellId', 'CellIndex', 'Mitotic state', 'Gene', 'Protein', 'ProteinDisplayName', 'Structure', 'StructureDisplayName', 'StructureShortName', 'ProteinId/Name', 'StructureId/Name', 'save_reg_path', 'ControlMask']]}")
    print(f"{intIdx} ({ind}): {df_train_filtered_nocontrols.loc[ind]['save_reg_path']}")

    #print(f'ref_x.shape = {ref_x.shape}')
    #print(f'target_x.shape = {target_x.shape}')

    objFig, (objAx1, objAx2) = plt.subplots(1, 2, figsize = (12, 8))
    plt.suptitle(f"{intIdx}: {df_train_filtered_nocontrols.loc[ind]['save_reg_path']}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    imshow_subplot(ref_x, objAx1, scale_channels=True, scale_global=False)
    imshow_subplot(target_x[0], objAx2, scale_channels=True, scale_global=False)


# ## Filter by cell features (leaving out control cells)

# In[179]:


objFig, objAxes = plt.subplots(1, 2, figsize=(20, 5))

_ = df_embeddings_plus_meta_train_pretty_names['Cell height'].hist(ax=objAxes[0])
_ = objAxes[0].set_title('Histogram of cell height', fontsize=20)
_ = objAxes[0].set_xlabel('Cell height', fontsize=14)
_ = objAxes[0].set_ylabel('Num cells', fontsize=14)

_ = df_embeddings_plus_meta_train_pretty_names['Cell volume'].hist(ax=objAxes[1])
_ = objAxes[1].set_title('Histogram of cell volume', fontsize=20)
_ = objAxes[1].set_xlabel('Cell volume', fontsize=14)
_ = objAxes[1].set_ylabel('Num cells', fontsize=14)


# In[294]:


mitotic_state = 'M0'
#structure = 'Lysosome'
cell_height = 20
#cell_height = 50
operator = '<='
#operator = '>='
sort_dir=False

#df_train_filtered = df_embeddings_plus_meta_train_pretty_names.query(f"`Mitotic state` == @mitotic_state & Structure == @structure & `Cell height` {operator} @cell_height")
df_train_filtered = df_embeddings_plus_meta_train_pretty_names.query(f"`Mitotic state` == @mitotic_state & `Cell height` {operator} @cell_height")
df_train_filtered_nocontrols = df_train_filtered[df_train_filtered['ControlMask'] == False].sort_values('Cell height', ascending=sort_dir)

df_train_filtered_nocontrols[['UnsplitCsvIndex', 'CellId', 'Mitotic state', 'Structure', 'Cell height', 'save_reg_path']]


# In[297]:


#ind = df_train_filtered_nocontrols.index[-1]
ind = 24047

ref_x = dp_ref.get_sample('train', [ind])
target_x = dp_target.get_sample('train', [ind])

#print(f"{df_train_filtered_nocontrols.loc[ind][['UnsplitCsvIndex', 'CellId', 'CellIndex', 'Mitotic state', 'Gene', 'Protein', 'ProteinDisplayName', 'Structure', 'StructureDisplayName', 'StructureShortName', 'ProteinId/Name', 'StructureId/Name', 'save_reg_path', 'ControlMask']]}")
print(f"{ind}: {df_train_filtered_nocontrols.loc[ind]['save_reg_path']}")

#print(f'ref_x.shape = {ref_x.shape}')
#print(f'target_x.shape = {target_x.shape}')

plt.figure(figsize=(6, 8))
imshow(ref_x, scale_channels=True, scale_global=False)
plt.figure(figsize=(6, 8))
imshow(target_x[0], scale_channels=True, scale_global=False)


# In[295]:


iterIdx = iter(df_train_filtered_nocontrols.index)
print(f'{len(df_train_filtered_nocontrols.index)}')


# In[296]:


print(f'image_parent = {dp_ref.image_parent}, {dp_target.image_parent}')

for intIdx in np.arange(10):
    #ind = df_train_filtered_nocontrols.index[-1]
    ind = next(iterIdx)

    ref_x = dp_ref.get_sample('train', [ind])
    target_x = dp_target.get_sample('train', [ind])

    #print(f"{df_train_filtered_nocontrols.loc[ind][['UnsplitCsvIndex', 'CellId', 'CellIndex', 'Mitotic state', 'Gene', 'Protein', 'ProteinDisplayName', 'Structure', 'StructureDisplayName', 'StructureShortName', 'ProteinId/Name', 'StructureId/Name', 'save_reg_path', 'ControlMask']]}")
    print(f"{intIdx} ({ind}): {df_train_filtered_nocontrols.loc[ind]['save_reg_path']}")

    #print(f'ref_x.shape = {ref_x.shape}')
    #print(f'target_x.shape = {target_x.shape}')

    objFig, (objAx1, objAx2) = plt.subplots(1, 2, figsize = (12, 8))
    plt.suptitle(f"{intIdx}: {df_train_filtered_nocontrols.loc[ind]['save_reg_path']}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    imshow_subplot(ref_x, objAx1, scale_channels=True, scale_global=False)
    imshow_subplot(target_x[0], objAx2, scale_channels=True, scale_global=False)


# ## Load the drug dataproviders

# In[51]:


from integrated_cell.utils import load_drug_data_provider

dp_ref_drugs = load_drug_data_provider(dp_ref, args_ref)
dp_target_drugs = load_drug_data_provider(dp_target, args_target)

#sanity check
assert dp_ref_drugs.__len__('test') == dp_target_drugs.__len__('test')


# In[235]:


dp_ref_drugs.csv_data[
    [
        'index', 
        'CellId', 
        'CellIndex', 
        'drug_label', 
        'drug_name', 
        'concentration', 
        'save_dir', 
        'save_flat_reg_path', 
        'save_flat_proj_reg_path', 
        'save_feats_path', 
        'save_reg_path', 
        'save_reg_path_flat', 
        'save_reg_path_flat_proj', 
    ]
]


# ## Filter by drug name, concentration, and structure (no control cells to filter out)

# In[245]:


print(f"Drug names = {dp_ref_drugs.csv_data['drug_name'].unique()}")
print(f"Concentrations = {dp_ref_drugs.csv_data['concentration'].unique()}")
print(f"Treatment group = {dp_ref_drugs.csv_data['treatment_group'].unique()}")
print(f"Structures = {dp_ref_drugs.csv_data['Structure'].unique()}")


# In[244]:


dp_ref_drugs.csv_data.to_csv('/allen/aics/modeling/caleb/data/dp_ref_drugs.csv')
dp_target_drugs.csv_data.to_csv('/allen/aics/modeling/caleb/data/dp_target_drugs.csv')


# In[266]:


#drug = 'Paclitaxel'
drug = 'Brefeldin'
concentration = 5.0
#structure = 'Microtubules'
structure = 'Golgi'

df_train_drugs_filtered = dp_ref_drugs.csv_data.query("drug_name == @drug & concentration == @concentration & Structure == @structure")
# No need to filter out control cells since there are no control cells in the drug data providers
#df_train_drugs_filtered_nocontrols = df_train_drugs_filtered[~df_train_drugs_filtered['StructureDisplayName'].str.contains('Control')]

df_train_drugs_filtered[['CellId', 'drug_name', 'concentration', 'Structure', 'save_reg_path']]


# ## Visualize cell images

# In[271]:


print(f'image_parent = {dp_ref_drugs.image_parent}, {dp_target_drugs.image_parent}')

ind = df_train_drugs_filtered.index[15]

ref_x_drugs = dp_ref_drugs.get_sample('test', [ind])
target_x_drugs = dp_target_drugs.get_sample('test', [ind])

#print(f"{df_train_drugs_filtered.loc[ind][['index', 'CellId', 'CellIndex', 'drug_label', 'drug_name', 'concentration', 'Gene', 'Protein', 'ProteinDisplayName', 'Structure', 'StructureDisplayName', 'StructureShortName', 'ProteinId/Name', 'StructureId/Name', 'save_dir', 'save_reg_path']]}")
print(f"{df_train_drugs_filtered.loc[ind]['save_reg_path']}")

#print(f'ref_x_drugs.shape = {ref_x_drugs.shape}')
#print(f'target_x_drugs.shape = {target_x_drugs.shape}')

plt.figure(figsize=(6, 8))
imshow(ref_x_drugs, scale_channels=True, scale_global=False)
plt.figure(figsize=(6, 8))
imshow(target_x_drugs[0], scale_channels=True, scale_global=False)


# In[268]:


iterIdx = iter(df_train_drugs_filtered.index)
print(f'{len(df_train_drugs_filtered.index)}')


# In[270]:


print(f'image_parent = {dp_ref_drugs.image_parent}, {dp_target_drugs.image_parent}')

for intIdx in np.arange(10):
    #ind = df_train_drugs_filtered.index[0]
    ind = next(iterIdx)

    ref_x_drugs = dp_ref_drugs.get_sample('test', [ind])
    target_x_drugs = dp_target_drugs.get_sample('test', [ind])

    #print(f"{df_train_drugs_filtered.loc[ind][['index', 'CellId', 'CellIndex', 'drug_label', 'drug_name', 'concentration', 'Gene', 'Protein', 'ProteinDisplayName', 'Structure', 'StructureDisplayName', 'StructureShortName', 'ProteinId/Name', 'StructureId/Name', 'save_dir', 'save_reg_path']]}")
    print(f"{intIdx}: {df_train_drugs_filtered.loc[ind]['save_reg_path']}")

    #print(f'ref_x_drugs.shape = {ref_x_drugs.shape}')
    #print(f'target_x_drugs.shape = {target_x_drugs.shape}')

    objFig, (objAx1, objAx2) = plt.subplots(1, 2, figsize = (12, 8))
    plt.suptitle(f"{intIdx}: {df_train_drugs_filtered.loc[ind]['save_reg_path']}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    imshow_subplot(ref_x_drugs, objAx1, scale_channels=True, scale_global=False)
    imshow_subplot(target_x_drugs[0], objAx2, scale_channels=True, scale_global=False)


# In[ ]:




