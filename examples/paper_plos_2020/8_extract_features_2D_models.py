#!/usr/bin/env python
# coding: utf-8

# # 2D model feature extraction

# ## Display a summary of the 2D models

# In[ ]:


import os
import pickle
import pandas as pd

import numpy as np
import torch

import features_lib as flib
from integrated_cell import model_utils, utils


# In[ ]:


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

# Parent directory of where the extracted features will be saved
feats_parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/results/feats_caleb/'

intNumCells = 5  # Set to < 0 to select the entire test set

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
# TODO: Allow the user to specify these models by listing intensity_norm and beta, and
#       get the model paths from dfCSV_2DModels above
model_dirs = [
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_298/',  # norm = 0, beta = 0.010
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_299/',  # norm = 1, beta = 0.010
    
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_312/',  # norm = 0, beta = 0.296
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_313/',  # norm = 1, beta = 0.296
    
    #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_329/',  # norm = 1, beta = 0.623
    
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_330/',  # norm = 0, beta = 0.663
    '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_331/',  # norm = 1, beta = 0.663
    
    # All generated cells look the same since beta is too high, not a good model to use
    #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_378/',  # norm = 0, beta = 0.990
    #'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_379/',  # norm = 1, beta = 0.990
]

strEmbeddingsParentPath = '/allen/aics/modeling/caleb/data/'

debug = False


# In[ ]:


if (not flib.fnIsBatchMode()):
    # Automatically reload modules before code execution
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


if (not flib.fnIsBatchMode()):
    # Set plotting style
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ### Create a data structure of the loaded models' properties

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
    
    

datStart = flib.fnNow()
print(f'Started on {flib.fnGetDatetime(datStart)}')
print()

data_list = list()
for i, model_dir in enumerate(model_dirs):
    print(model_dir)
    
    # do model selection based on validation data
    model_summaries = flib.get_embeddings_for_dir(model_dir, parent_dir, use_current_results = False, mode='validate')

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
    model_summaries = flib.get_embeddings_for_dir(model_dir, parent_dir, use_current_results = False, mode = "test", suffixes=[best_suffix])
    
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

datEnd = flib.fnNow()
print()
print(f'Ended on {flib.fnGetDatetime(datEnd)}')

datDuration = datEnd - datStart
print(f'datDuration = {datDuration}')
print()


# In[ ]:


# Display some information about the loaded models
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


if debug: print(f'Num. models = {len(im_scores_gen)}\n{im_scores_gen[0]}')


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


# Display some information about the loaded models in a dataframe
df_master[['beta', 'intensity_norm', 'suffix', 'model_dir']].sort_values(['beta', 'intensity_norm'])
#df_master.to_csv('~/df_master.csv', index = False)


# ## Do feature calculation for some subset of models

# In[ ]:


from tqdm import tqdm

# All = (30, 4), single = (10, 4)
figsize_cells_real = (30, 4) if seg_method_real == 'all' else (10, 4)
figsize_cells_gen = (30, 4) if seg_method_gen == 'all' else (10, 4)

datStart = flib.fnNow()
print(f'Started on {flib.fnGetDatetime(datStart)}')
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
            flib.save_feats(
                im, 
                save_real_feat_path, 
                seg_method=seg_method_real, 
                mask_intensity_features=mask_intensity_features_real, 
                save_imgs=save_imgs, 
                figsize_hist=figsize_hist, 
                figsize_cells=figsize_cells_real, 
                debug=debug
            )

    feature_path_dict[intensity_norm] = {}
    feature_path_dict[intensity_norm]['real'] = flib.load_feats(save_real_feats_paths)
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
        
        strEmbeddingsFullFilename = flib.fnGenUnsortedEmbeddingsPath(strEmbeddingsParentPath, strModelDir, strRefSuffix)
        dctUnsortedEmbeddings = flib.fnLoadUnsortedEmbeddings(strEmbeddingsFullFilename)
        z_test = dctUnsortedEmbeddings['test'][cell_idx, :]
        print(f'    z_test.shape = {z_test.shape}')
        
        z_rnd_kld = flib.fnGenRandomZ(strEmbeddingsParentPath, strModelDir, strRefSuffix, argBatchSize=n_dat)
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

                    flib.save_feats(
                        im, 
                        save_path, 
                        seg_method=seg_method_gen, 
                        mask_intensity_features=mask_intensity_features_gen, 
                        save_imgs=save_imgs, 
                        figsize_hist=figsize_hist, 
                        figsize_cells=figsize_cells_gen, 
                        debug=debug
                    )

            #beta = df_norm['beta'].iloc[i]
            #print(f'    beta = {beta}')

            #feature_path_dict[intensity_norm]['gen'][beta][key] = load_feats(save_gen_feats_paths)
            
            feature_path_dict[intensity_norm]['gen'][beta][key] = {}
            feature_path_dict[intensity_norm]['gen'][beta][key]['embeddings'] = z_tmp.cpu().detach().numpy()
            feature_path_dict[intensity_norm]['gen'][beta][key]['features'] = flib.load_feats(save_gen_feats_paths)
            
with open(all_feats_save_path, "wb") as f:
    pickle.dump(feature_path_dict, f)

datEnd = flib.fnNow()
print()
print(f'Ended on {flib.fnGetDatetime(datEnd)}')

datDuration = datEnd - datStart
print(f'datDuration = {datDuration}')
print()


# In[ ]:




