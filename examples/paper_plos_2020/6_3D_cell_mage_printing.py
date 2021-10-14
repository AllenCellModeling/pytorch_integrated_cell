#!/usr/bin/env python
# coding: utf-8

# ## Combined reference and target images for image making

# In[1]:


import json
import integrated_cell
from integrated_cell import model_utils, utils
import os
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

from integrated_cell.utils.plots import tensor2im, imshow

gpu_ids = [7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ID) for ID in gpu_ids])
if len(gpu_ids) == 1:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()
    
#model_dir = '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:27:04'  # CC
model_dir = '/allen/aics/modeling/ic_data/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:27:04'  # CC
#parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/'  # CC
parent_dir = '/allen/aics/modeling/ic_data/results/integrated_cell/'  # CC
suffix = '_94544'    

networks, dp_ref, args_ref = utils.load_network_from_dir(model_dir, parent_dir, suffix=suffix)  # CC

ref_enc = networks['enc']
ref_dec = networks['dec']

#parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/'  # CC
parent_dir = '/allen/aics/modeling/ic_data/results/integrated_cell/'  # CC
#model_dir = "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/"  # CC
model_dir = "/allen/aics/modeling/ic_data/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/"  # CC
suffix = '_93300'

networks, dp_target, args_target = utils.load_network_from_dir(model_dir, parent_dir, suffix=suffix)  # CC
    
target_enc = networks['enc']
target_dec = networks['dec']


# NOTE: Not used, removed by CC
#results_dir = '{}/results/ref_target_images/'.format(parent_dir)  # CC
#if not os.path.exists(results_dir):
#    os.makedirs(results_dir)
#    
#print("Results dir: {}".format(results_dir))
#
#save_dir = results_dir


# In[2]:


from aicsimageio.writers import OmeTiffWriter

def im_write(im, path):
    im = im.cpu().detach().numpy().transpose(3,0,1,2)
    
    # CC
    with OmeTiffWriter(path, overwrite_file=True) as writer:
        writer.save(im)


# In[3]:


from integrated_cell.networks.ref_target_autoencoder import Autoencoder


mode = 'test'
dp = dp_target
u_classes, class_inds = np.unique(dp.get_classes(np.arange(0, dp.get_n_dat(mode)), mode), return_inverse=True)
u_class_names = dp.label_names[u_classes]

ae = Autoencoder(ref_enc, ref_dec, target_enc, target_dec)
ae.train(False)
ae = ae.cuda()


# In[4]:


target, labels, ref = dp.get_sample(mode)
label_onehot = utils.index_to_onehot(labels, len(u_classes)).cuda()
target = target.cuda()
ref = ref.cuda()

with torch.no_grad():
    target_hat, ref_hat = ae(target, ref, label_onehot)

im = torch.cat([ref[:,[0]], target, ref[:,[1]]], 1)
im_hat = torch.cat([ref_hat[:,[0]], target_hat, ref_hat[:,[1]]], 1)


# ## Save Real and Autoencoded Images

# In[5]:


import integrated_cell.utils.plots as plots
import tqdm

#ae_dir = "./images/ae/"  # CC
ae_dir = f"{parent_dir}/notebook_6/images/ae/"  # CC
if not os.path.exists(ae_dir):
    os.makedirs(ae_dir)        
        
c = 0
for i in tqdm.tqdm(range(10)):
        
    target, labels, ref = dp.get_sample(mode)
    label_onehot = utils.index_to_onehot(labels, len(u_classes)).cuda()
    target = target.cuda()
    ref = ref.cuda()

    with torch.no_grad():
        target_hat, ref_hat = ae(target, ref, label_onehot)

    im = torch.cat([ref[:,[0]], target, ref[:,[1]]], 1)
    im_hat = torch.cat([ref_hat[:,[0]], target_hat, ref_hat[:,[1]]], 1)

    for i, [im_, im_hat_, structure_type] in enumerate(zip(im, im_hat, u_class_names[labels])):
        im_write(im_, "{}/{}_im{}_real.tiff".format(ae_dir, structure_type, c))  # CC
        im_write(im_hat_, "{}/{}_im{}_ae.tiff".format(ae_dir, structure_type, c))  # CC
        
        c+=1


# In[6]:


structures_to_gen = ["Mitochondria", 'Nuclear envelope', 'Tight junctions']

structure_ids_to_gen = np.stack([np.where(u_class_names == structure)[0] for structure in structures_to_gen])

structure_to_gen_ids = [np.where(class_inds == structure_id)[0] for structure_id in structure_ids_to_gen]
    
#i chose these
structure_to_gen_inds = [1, 0, 0]

im_ids = [structure_to_gen[ind] for structure_to_gen, ind in zip(structure_to_gen_ids, structure_to_gen_inds)]


# In[7]:


torch.zeros([9, len(u_classes)]).float().cuda()


# In[9]:


from matplotlib.image import imsave as imsave

#gen_dir = "./images/gen/"  # CC
gen_dir = f"{parent_dir}/notebook_6/images/gen/"  # CC
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)   
    
n_structures = len(structures_to_gen)
n_imgs_per_structure = 3

n_cols = int(n_structures*n_imgs_per_structure)

target, label, ref = dp.get_sample(mode, im_ids)
label_onehot = utils.index_to_onehot(label, len(u_classes)).cuda()

target = target.cuda()
ref = ref.cuda()

im = torch.cat([ref[:,[0]], target, ref[:,[1]]], 1)

reals = list()
for i in range(n_structures):
    real = plots.tensor2im(im[[i]])
    imsave("{}/real_{}.png".format(gen_dir, structures_to_gen[i]), real)  # CC
    
    im_tmp = im[[i]]
    im_tmp[:, [0,2]] = 0
    real_no_ref = plots.tensor2im(im_tmp)
    
    imsave("{}/real_{}_no_ref.png".format(gen_dir, structures_to_gen[i]), real_no_ref)  # CC
    
    
    reals.append(plots.tensor2im(ref[[i]]))
    
imsave("{}/real_ref.png".format(gen_dir), np.hstack(reals))  # CC
    

#generate 9 cell and nuc images by passing in only labels into the AE
labels_onehot_dummy = torch.zeros([n_cols, len(u_classes)]).float().cuda()

with torch.no_grad():
    _, ref_gen = ae(target=None, ref=None, labels = labels_onehot_dummy)

gen_ref = np.hstack([plots.tensor2im(ref_gen[[i]]) for i in range(ref_gen.shape[0])])    

imsave("{}/gen_ref.png".format(gen_dir), gen_ref)  # CC
    
plt.figure(figsize=[20,20])
plt.imshow(gen_ref)
    
labels_gen = torch.cat([label_onehot[[i]].repeat([n_imgs_per_structure,1]) for i in range(n_structures)],0)

for i in range(n_structures):
    ref_tmp = ref[[i]].repeat([n_cols, 1, 1, 1, 1])
    
    target_gen, _ = ae(target=None, ref=ref_tmp, labels = labels_gen)
    
    gen_imgs = np.hstack([plots.tensor2im(target_gen[[i]], color_transform=[[1,1,0]]) for i in range(target_gen.shape[0])])
    
    imsave("{}/gen_{}.png".format(gen_dir, structures_to_gen[i]), gen_imgs)  # CC
    
    plt.figure(figsize=[20,20])
    plt.imshow(gen_imgs)
    plt.show()
    plt.close()


# In[10]:


for u_label in np.unique(dp_ref.labels):
    n_labels = np.sum(dp_ref.labels == u_label)
    label_name = dp_ref.label_names[u_label]
    print(f"{label_name}: {n_labels}")


# In[ ]:




