#!/usr/bin/env python
# coding: utf-8

<<<<<<< HEAD
# ### New GitHub branch

# In[ ]:


Repo:   AllenCellModeling/pytorch_integrated_cell
Branch: plos_paper_2021


# ### TODO

# In[ ]:


get_ipython().set_next_input('- Restart kernel and clear all outputs before pushing to github');get_ipython().run_line_magic('pinfo', 'github')
- Remove '# CC' once the noteboks have been confirmed to run with the new data locations


# ### Folders that need to be copied to a new location

# In[ ]:


# Copy log

New location: '/allen/aics/modeling/ic_data/'

"/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/"  # 403G (done)
"/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-31-21:48:56/"  # 380G (done)
"/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:27:04/"  # 370G (done)
"/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:23:27/"  # 370G (done)
       
'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae/2019-07-19-09:27:15/'  # 1.2T (done)

3 '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_*/'  # ? (in progress)

'/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'  # 364G (done)
'/raid/shared/ipp/scp_19_04_10/'  # 274G (done)


# ### Keywords to search for

# In[ ]:


- gregj
- rory
- save
- dump
- parent_dir
- results_dir
- os.makedirs

- asdf
- sorry
- swear words of your choice


# ### Notebook #1

=======
# In[ ]:


"/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/"  # 403G
"/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-31-21:48:56/"  # 380G
"/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:27:04/"  # 370G
"/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:23:27/"  # 370G
       
'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae/2019-07-19-09:27:15/'  # 1.2T

'/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_*/'  # 

'/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'  # 364G
'/raid/shared/ipp/scp_19_04_10/'  # 274G


>>>>>>> 54d4203e42f613adc5b6b9aa9fce50face40f1d9
# In[ ]:


# Notebook 1

parent_dir = "/allen/aics/modeling/gregj/results/integrated_cell/"  # CC
model_parent = '{}/test_cbvae_beta_ref'.format(parent_dir)  # CC
model_dirs = glob.glob('/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_beta_ref/job_*/')  # CC
        
save_dir = '{}/results'.format(model_parent)  # CC
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
results_dir = save_dir  # CC

def get_embeddings_for_dir(model_dir, parent_dir, use_current_results=False):
        model_summary_path = "{}/ref_model/embeddings_validate{}_summary.pth".format(model_dir, suffix)  # CC_write: writing notebook output to model path???
            with open(model_summary_path, "wb") as f:
                pickle.dump(model_summary, f)
        embeddings_path = "{}/ref_model/embeddings_{}{}.pth".format(model_dir, mode, suffix)  # CC
        embeddings = get_embeddings_for_model(suffix, model_dir, parent_dir, embeddings_path, use_current_results, mode = mode)

model_summaries = get_embeddings_for_dir(model_dir, parent_dir, use_current_results = False, mode='validate')  # CC
model_summaries = get_embeddings_for_dir(model_dir, parent_dir, use_current_results = False, mode = "test", suffixes=[best_suffix])  # CC

_, dp, _ = utils.load_network_from_dir(data_list[0]['model_dir'], parent_dir)  # CC
dp.image_parent = '/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'  # CC

plt.savefig('{}/model_selection_beta.png'.format(save_dir), bbox_inches='tight', dpi=90)  # CC
plt.savefig('{}/model_selection_beta_clean.png'.format(results_dir), bbox_inches='tight', dpi=90)  # CC

best_model = ''  # CC
if data['model_dir'] == "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae/2019-07-19-09:27:15/":  # CC
    best_model = i
    break

def save_feats(im, save_path):
    with open(save_path, "wb") as f:  # CC
        pickle.dump(feats, f)

feats_parent_dir = "{}/feats/".format(results_dir)  # CC
all_feats_save_path = "{}/all_feats.pkl".format(feats_parent_dir)  # CC

save_norm_parent = "{}/norm_{}".format(feats_parent_dir, intensity_norm)  # CC
if not os.path.exists(save_norm_parent):
    os.makedirs(save_norm_parent)

save_norm_feats = "{}/feats_test".format(save_norm_parent)  # CC
if not os.path.exists(save_norm_feats):
    os.makedirs(save_norm_feats)

networks, dp, args = utils.load_network_from_dir(df_norm['model_dir'].iloc[0], parent_dir, suffix=df_norm['suffix'].iloc[0])  # CC

save_real_feats_paths = ['{}/feat_{}.pkl'.format(save_norm_feats, i) for i in range(n_dat)]  # CC
save_feats(im, save_real_feat_path)

save_feats_dir = '{}/{}'.format(save_norm_parent, df_master['label'].iloc[i])  # CC
if not os.path.exists(save_feats_dir):
    os.makedirs(save_feats_dir)

save_gen_feats_paths = ['{}/feat_{}.pkl'.format(save_feats_dir, i) for i in range(n_dat)]  # CC
for j, save_path in tqdm(enumerate(save_gen_feats_paths)):
        save_feats(im, save_path)  # CC
        
with open(all_feats_save_path, "wb") as f:  # CC
    pickle.dump(feature_path_dict, f)    

all_feats_save_path = '/allen/aics/modeling/gregj/results/integrated_cell//test_cbvae_beta_ref/results/feats//all_feats.pkl'  # CC

plt.savefig('{}/features_and_betas.png'.format(results_dir), bbox_inches='tight', dpi=90)  # CC

gen_dir = f"{results_dir}/gen"  # CC
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)

gen_img_dir = f"{gen_dir}/{label}"  # CC
if not os.path.exists(gen_img_dir):
    os.makedirs(gen_img_dir)

networks, dp, args = utils.load_network_from_dir(row['model_dir'], parent_dir, suffix = row['suffix'])  # CC

scipy.misc.imsave(f'{gen_dir}/im_{j}.png', im)  # CC

scipy.misc.imsave(f'{results_dir}/im_generated.png', im_out)  # CC
scipy.misc.imsave(f'{results_dir}/im_sampled.png', im_real)  # CC
scipy.misc.imsave(f'{results_dir}/im_sampled.png', im_real)  # CC


<<<<<<< HEAD
# ### Notebook #2

=======
>>>>>>> 54d4203e42f613adc5b6b9aa9fce50face40f1d9
# In[ ]:


# Notebook 2

# NOTE: Had to change GPU ID

parent_dir = "/allen/aics/modeling/gregj/results/integrated_cell/"  # CC
model_parent = '{}/test_cbvae_avg_inten'.format(parent_dir)  # CC

# NOTE: Looks like these two lists were over-written and model selection was only done in the 2nd list???
model_dirs = ["/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/",
             "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-31-21:48:56/",
model_dirs = ["/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:27:04",
              "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:23:27",
              
def get_embeddings_for_dir(model_dir, parent_dir, use_current_results=False):
        model_summary_path = "{}/ref_model/embeddings_validate{}_summary.pth".format(model_dir, suffix)  # CC_write: writing notebook output to model path???
            with open(model_summary_path, "wb") as f:
                pickle.dump(model_summary, f)
            embeddings_path = "{}/ref_model/embeddings_validate{}.pth".format(model_dir, suffix)  # CC
            embeddings = get_embeddings_for_model(suffix, model_dir, parent_dir, embeddings_path, use_current_results)  # CC

def get_embeddings_for_model(suffix, model_dir, parent_dir, save_path, use_current_results):
        networks, dp, args = utils.load_network_from_dir(model_dir, parent_dir, suffix = suffix)  # CC

def load_network_from_dir(
        dp_kwargs["save_path"] = dp_kwargs["save_path"].replace("./", parent_dir)
        net_kwargs[net_name]["save_path"] = net_kwargs[net_name]["save_path"].replace(
            "./", parent_dir
        )
# save_path = './test_cbvae_3D_avg_inten/data_rescaled_ref.pyt', parent_path = '/allen/aics/modeling/gregj/results/integrated_cell/'
# save_path = './test_cbvae_3D_avg_inten/2019-11-27-22:23:27/ref_model/enc.pth', parent_path = ''/allen/aics/modeling/gregj/results/integrated_cell/'

save_dir = '{}/results'.format(model_parent)  # CC
dp.image_parent = '/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'  # CC (before reassignment = '/raid/shared/ipp/scp_19_04_10/')

plt.savefig('{}/model_selection_beta.png'.format(save_dir), bbox_inches='tight', dpi=90)  #CC_write

im_out_path = '{}/ref_model/progress_{}.png'.format(data['model_dir'], int(epoch_num))
# Sample model_dir = /allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:23:27

best_model = 'asdfasdfasdf'  # What's the deal with this?
if data['model_dir'] == "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae/2019-07-19-09:27:15/":  # What's the deal with this hard-coding?
        best_model = i
        break

# NOTE: Are the best models selected the same ones used in notebook #7?


caleb.chan@dgx-aics-dcp-001:~/modeling_root/ic_data$ tree
.
└── results
    ├── integrated_cell
    │   ├── test_cbvae -> /allen/aics/modeling/gregj/results/integrated_cell/test_cbvae/
    │   ├── test_cbvae_3D_avg_inten -> /allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/
    │   └── test_cbvae_avg_inten
    │       └── results
    │           └── model_selection_beta.png
    └── ipp
        ├── scp_19_04_10 -> /allen/aics/modeling/gregj/results/ipp/scp_19_04_10/
        └── scp_19_04_10_raid -> /raid/shared/ipp/scp_19_04_10/


<<<<<<< HEAD
# ### Notebook #3

=======
>>>>>>> 54d4203e42f613adc5b6b9aa9fce50face40f1d9
# In[ ]:


# Notebook #3

parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/'  # CC
model_dir = "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/"  # CC

networks, dp, args = utils.load_network_from_dir(model_dir, parent_dir, suffix=suffix)  # CC

# NOTE: Is it good practice to write analyses results into model_dir?
results_dir = '{}/results/kl_demo{}/'.format(model_dir, suffix)  # CC
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

dp.image_parent = '/allen/aics/modeling/gregj/results/ipp/scp_19_04_10/'  # CC

save_dir = results_dir  # CC

# NOTE: Should we write to somewhere else other than where the notebook is?
with OmeTiffWriter('real.tiff', overwrite_file=True) as writer:
    writer.save(im_proc(im_real[ind]))  # CC

with OmeTiffWriter('gen.tiff', overwrite_file=True) as writer:
    writer.save(im_proc(im_real[ind]))  # CC

# NOTE: Hard-coding font folder?
import integrated_cell.utils.target.plots as plots
im_classes_real, im_classes_gen = plots.summary_images(dp, enc, dec)  # CC
def summary_images(...)
    font = ImageFont.truetype(
        os.path.dirname(__file__) + "/../../../etc/arial.ttf", font_size
    )

matplotlib.image.imsave('{}/im_summary.png'.format(results_dir), im_classes)  # CC

matplotlib.image.imsave('{}/im_summary_v2.png'.format(results_dir), im_classes_v2)  # CC
matplotlib.image.imsave('{}/im_summary_v2_remaining.png'.format(results_dir), im_classes_v2_remaining)  # CC

# NOTE: Should we write to somewhere else other than where the notebook is?
save_dir = './demo_imgs/'  # CC
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

tifffile.imsave("{}/test_{}.tiff".format(save_dir, i), im_out)  # CC

matplotlib.image.imsave('{}/im_sampled_struct.png'.format(results_dir), im_struct)  # CC
matplotlib.image.imsave('{}/im_sampled_struct_hat.png'.format(results_dir), im_struct_hat)  # CC

embeddings_path = '{}/embeddings.pth'.format(results_dir)  # CC
if not os.path.exists(embeddings_path):
    embeddings = get_latent_embeddings(enc, dec, dp, recon_loss, batch_size = 32)
    torch.save(embeddings, embeddings_path)

else:
    embeddings = torch.load(embeddings_path)

<<<<<<< HEAD
# TODO
=======
>>>>>>> 54d4203e42f613adc5b6b9aa9fce50face40f1d9
from lkaccess import LabKey  # CC
lk = LabKey(host="aics")
mito_data = lk.select_rows_as_list(
   schema_name="processing",
   query_name="MitoticAnnotation",
   sort="MitoticAnnotation",
   columns=["CellId", "MitoticStateId", "MitoticStateId/Name", "Complete"]
)

def embedding_variation(embeddings, figsize = (8, 4), save_path = None):  # CC
if save_path is not None:
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)  # CC
    plt.close()

plt.savefig('{}/dimension_klds.png'.format(results_dir), bbox_inches='tight', dpi=dpi)  # CC
plt.savefig('{}/dimension_target.png'.format(results_dir), bbox_inches='tight', dpi=dpi)  # CC

save_klds_path = "{}/klds.pkl".format(results_dir)  # CC
with open(save_klds_path, 'wb') as f:
    pickle.dump([klds_struct, target_sorted_inds], f)
    
plt.savefig('{}/dimension_klds_struct.png'.format(results_dir), bbox_inches='tight', dpi=dpi)  # CC
plt.savefig('{}/dimension_klds_combined.png'.format(results_dir), bbox_inches='tight', dpi=dpi)  # CC

df_data.to_csv('{}/stats.csv'.format(save_dir))  # CC
embeddings_mat.to_csv(filename)  # CC

plt.savefig('{}/p_value_distributions.png'.format(results_dir), dpi=dpi)  # CC
plt.savefig('{}/p_value_distributions_bar.png'.format(results_dir), dpi=dpi, bbox_inches='tight') # CC
plt.savefig('{}/p_value_elbo_density.png'.format(results_dir), dpi=dpi, bbox_inches='tight')  # CC
plt.savefig('{}/p_value_elbo_scatter.png'.format(results_dir), dpi=dpi, bbox_inches='tight')  # CC

p_value_elbo_dir = '{}/p_value_elbo_scatter'.format(results_dir)  # CC
if not os.path.exists(p_value_elbo_dir):
    os.makedirs(p_value_elbo_dir)

plt.savefig('{}/{}.png'.format(p_value_elbo_dir, label_to_print.replace('(', '').replace(')', '')), dpi=90, bbox_inches='tight')  # CC

caleb.chan@dgx-aics-dcp-001:~/modeling_root/ic_data$ tree
.
└── results
    ├── integrated_cell
    │   ├── notebook_3
    │   │   ├── demo_imgs
    │   │   │   ├── gen.tiff
    │   │   │   ├── real.tiff
    │   │   │   ├── stats.csv
    │   │   │   ├── test_0.tiff
    │   │   │   ├── test_10.tiff
    │   │   │   ├── test_11.tiff
    │   │   │   ├── test_12.tiff
    │   │   │   ├── test_13.tiff
    │   │   │   ├── test_14.tiff
    │   │   │   ├── test_15.tiff
    │   │   │   ├── test_16.tiff
    │   │   │   ├── test_17.tiff
    │   │   │   ├── test_18.tiff
    │   │   │   ├── test_19.tiff
    │   │   │   ├── test_1.tiff
    │   │   │   ├── test_20.tiff
    │   │   │   ├── test_21.tiff
    │   │   │   ├── test_22.tiff
    │   │   │   ├── test_23.tiff
    │   │   │   ├── test_24.tiff
    │   │   │   ├── test_25.tiff
    │   │   │   ├── test_26.tiff
    │   │   │   ├── test_27.tiff
    │   │   │   ├── test_28.tiff
    │   │   │   ├── test_29.tiff
    │   │   │   ├── test_2.tiff
    │   │   │   ├── test_30.tiff
    │   │   │   ├── test_31.tiff
    │   │   │   ├── test_3.tiff
    │   │   │   ├── test_4.tiff
    │   │   │   ├── test_5.tiff
    │   │   │   ├── test_6.tiff
    │   │   │   ├── test_7.tiff
    │   │   │   ├── test_8.tiff
    │   │   │   └── test_9.tiff
    │   │   └── kl_demo_2019-10-22-15:24:09_93300
    │   │       ├── dimension_klds_combined.png
    │   │       ├── dimension_klds.png
    │   │       ├── dimension_klds_struct.png
    │   │       ├── dimension_target.png
    │   │       ├── embeddings.pth
    │   │       ├── embeddings_target_mu_test.csv
    │   │       ├── embeddings_target_sigma_test.csv
    │   │       ├── im_sampled_struct_hat.png
    │   │       ├── im_sampled_struct.png
    │   │       ├── im_summary.png
    │   │       ├── im_summary_v2.png
    │   │       ├── im_summary_v2_remaining.png
    │   │       ├── klds.pkl
    │   │       ├── p_value_distributions_bar.png
    │   │       ├── p_value_distributions.png
    │   │       ├── p_value_elbo_density.png
    │   │       ├── p_value_elbo_scatter
    │   │       │   └── Actin filaments.png
    │   │       └── p_value_elbo_scatter.png
    │   ├── test_cbvae -> /allen/aics/modeling/gregj/results/integrated_cell/test_cbvae/
    │   ├── test_cbvae_3D_avg_inten -> /allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/
    │   └── test_cbvae_avg_inten
    │       └── results
    │           └── model_selection_beta.png
    └── ipp
        ├── scp_19_04_10 -> /allen/aics/modeling/gregj/results/ipp/scp_19_04_10/
        └── scp_19_04_10_raid -> /raid/shared/ipp/scp_19_04_10/


<<<<<<< HEAD
# ### Notebook #6

=======
>>>>>>> 54d4203e42f613adc5b6b9aa9fce50face40f1d9
# In[ ]:


# Notebook #6

model_dir = '/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-11-27-22:27:04'  # CC
parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/'  # CC
networks, dp_ref, args_ref = utils.load_network_from_dir(model_dir, parent_dir, suffix=suffix)  # CC

parent_dir = '/allen/aics/modeling/gregj/results/integrated_cell/'  # CC
model_dir = "/allen/aics/modeling/gregj/results/integrated_cell/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/"  # CC
networks, dp_target, args_target = utils.load_network_from_dir(model_dir, parent_dir, suffix=suffix)  # CC

# NOTE: Not used
results_dir = '{}/results/ref_target_images/'.format(parent_dir)  # CC
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# NOTE: Not used
save_dir = results_dir

def im_write(im, path):
    im = im.cpu().detach().numpy().transpose(3,0,1,2)
    
    with OmeTiffWriter(path, overwrite_file=True) as writer:
        writer.save(im)
        
ae_dir = "./images/ae/"  # CC
if not os.path.exists(ae_dir):
    os.makedirs(ae_dir)        
        
im_write(im_, "{}/{}_im{}_real.tiff".format(ae_dir, structure_type, c))  # CC
im_write(im_hat_, "{}/{}_im{}_ae.tiff".format(ae_dir, structure_type, c))  # CC

gen_dir = "./images/gen/"  # CC
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)   

imsave("{}/real_{}.png".format(gen_dir, structures_to_gen[i]), real)  # CC
imsave("{}/real_{}_no_ref.png".format(gen_dir, structures_to_gen[i]), real_no_ref)
imsave("{}/real_ref.png".format(gen_dir), np.hstack(reals))
imsave("{}/gen_ref.png".format(gen_dir), gen_ref)
imsave("{}/gen_{}.png".format(gen_dir, structures_to_gen[i]), gen_imgs)

