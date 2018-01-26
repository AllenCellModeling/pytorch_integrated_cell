#######    
### This function prints off the most likely predicted 
### channels for each of the cells in our dataset
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
import scipy.misc

import model_utils

from aicsimage.io import omeTifWriter
from imgToProjection import imgtoprojection
import PIL.Image

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pdb
from tqdm import tqdm


import arc_walk

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='gpu id')
parser.add_argument('--interp_method',default='arc', help='interp method')
parser.add_argument('--overwrite', type=bool, default=False, help='overwrite')
parser.add_argument('--n_steps_per', type=int, default=10, help='number of samples between mitosis states')
parser.add_argument('--suffix', type=str, default='', help='suffix on save dir')
args = parser.parse_args()

model_dir = args.parent_dir + os.sep + 'struct_model'
ref_dir = args.parent_dir + os.sep + 'ref_model'

mito_file = args.parent_dir + os.sep + 'data_jobs_out_mitotic_annotations.csv'
df_mito = pd.read_csv(mito_file)
df_mito = df_mito[['inputFolder', 'inputFilename', 'outputThisCellIndex', 'MitosisLabel']]


save_dir = args.parent_dir + os.sep + 'analysis' + os.sep + 'latent_walk_mitosis_' + args.interp_method + args.suffix+ os.sep
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
        

# logger_file = '{0}/logger_tmp.pkl'.format(model_dir)
opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))
print(opt)

opt.gpu_ids = args.gpu_ids

torch.manual_seed(opt.myseed)
torch.cuda.manual_seed(opt.myseed)
np.random.seed(opt.myseed)

dp = model_utils.load_data_provider(opt.data_save_path, opt.imdir, opt.dataProvider)
df_data = dp.csv_data


df_data = df_data.merge(df_mito, on=['inputFolder', 'inputFilename', 'outputThisCellIndex'], how='left')
df_data = df_data.rename(columns = {'MitosisLabel_y': 'MitosisLabel'})

df_data_labeled = df_data[~np.isnan(df_data['MitosisLabel'])]
labels = df_data_labeled['MitosisLabel']

# labels[labels > 2] = 3


ulabels = np.unique(labels)

embeddings_shape = model_utils.load_embeddings(ref_dir + os.sep + 'embeddings.pkl')

use_train_or_test = 'test'

df_train = df_data.iloc[dp.data[use_train_or_test]['inds']]
# df_train['MitosisLabel'][df_train['MitosisLabel']>2] = 3

positions = list()
for label in ulabels:
    label_inds = np.where(label == df_train['MitosisLabel'])
    
    embeddings = embeddings_shape[use_train_or_test][label_inds].numpy()
    D = squareform(pdist(embeddings, metric='cityblock'))
    positions.append(embeddings[np.argmin(np.sum(D, axis=0))])
#     positions.append(np.mean(embeddings,axis=0))
positions = np.vstack(positions)

#######    
### Load REFERENCE MODEL
#######

opt.channelInds = [0, 1, 2]
dp.opts['channelInds'] = opt.channelInds
opt.nch = len(opt.channelInds)
        
opt.nClasses = dp.get_n_classes()
opt.nRef = opt.nlatentdim

models, _, _, _, opt = model_utils.load_model(opt.model_name, opt)

dec = models['dec']
dec.train(False)

models = None
optimizers = None

print('Done loading model.')

#######    
### Main Loop
#######



opt.batch_size = 400
gpu_id = opt.gpu_ids[0]

    
nclasses = dp.get_n_classes()
nref = opt.nRef
nlatent = opt.nlatentdim

nsamples = args.n_steps_per
interp_states =  list(range(0,5))

# interp_states = [0, 3]

positions = positions[interp_states]

from scipy.interpolate import interp1d

def interp(nsamples, y):
    xs = np.linspace(0, 1, num=nsamples)
    
    positions = np.zeros([nsamples, y.shape[1]])
    
    for i in range(0, y.shape[1]):
        positions[:,i] = interp1d([0,1], y[:,i])(xs)
    
    return positions



positions_final = list()
interp_positions_final = list()
interp_fract_final = list()

for i in range(0, len(positions)-1):
    if args.interp_method == 'linear':
        interp_pts = interp(nsamples, positions[i:i+2])[0:-1]
    if args.interp_method == 'arc':
        interp_pts = arc_walk.linspace_sph_pol(positions[i], positions[i+1], nsamples)[0:-1]
        
    interp_fract = np.linspace(0, 1, interp_pts.shape[0])    

#     if i != len(positions)-2:
#         interp_pts = interp_pts[0:-1]
#         interp_fract = interp_fract[0:-1]

    positions_final.append(interp_pts)
    interp_fract_final.append(interp_fract)
    interp_positions_final.append(np.repeat([interp_states[i:i+2]], interp_pts.shape[0], axis=0))    
    
positions_final = np.concatenate(positions_final)
interp_positions_final = np.concatenate(interp_positions_final)
interp_fract_final = np.concatenate(interp_fract_final)    
    
data = np.hstack([interp_positions_final, np.expand_dims(interp_fract_final,1)])    

df_states = pd.DataFrame(data, columns=['source state', 'target state', 'fract interp'])

    
nframes = len(positions_final)    
    

    
for i in range(0, nframes):
    path = './{0}/step_{1}.png'.format(save_dir, int(i));
    if os.path.exists(path): os.remove(path)
        
    
def tensor2img(img, do_half=True, proj_all=True):
#     colormap = 'hsv'
#     colors = plt.get_cmap(colormap)(np.linspace(0, 1, img.size()[1]+1))
    
    colors = [[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]]
    
    # pdb.set_trace()
    
    img = img.numpy()
    im_out = list()
    for i in range(0, img.shape[0]):
        
        img_tmp = img[i]
        
        for j in range(0, len(img_tmp)):
            img_tmp[j] = img_tmp[j]/np.max(img_tmp[j])
        
        img_tmp = np.swapaxes(img_tmp, 1,3)
        im_proj = imgtoprojection(img_tmp, proj_all=proj_all,  colors = colors, global_adjust=True)
        im_proj = np.swapaxes(im_proj, 0, 2)
        
        im_proj = np.flip(im_proj,0)
        im_proj = np.flip(im_proj,1)
        
        im_out.append(im_proj)
    
    
    if do_half:
        first_half = math.ceil(len(im_out)/2)
        last_half = math.floor(len(im_out)/2)

        if last_half != len(im_out)/2:
            im_out.append(np.zeros(im_out[0].shape))

        img1 = np.concatenate(im_out[0:first_half], 1)
        img2 = np.concatenate(im_out[first_half:], 1)

        img = np.concatenate([img1, img2], 0)
    else:
        img = np.concatenate(im_out[:],1)
        
        
    # if len(img.shape) == 3:
    #     img = np.expand_dims(img, 3)

    # for i in range(0, len(img)):
    #     img[i] = img[i]/np.max(img[i])
    
    # img = np.swapaxes(img, 2,3)
    # img = imgtoprojection(np.swapaxes(img, 1, 3), proj_all=True,  colors = colors, global_adjust=True)

    return img    
    
# init = Variable(torch.Tensor(1, nref).normal_().repeat(nclasses,1).cuda(gpu_id))

stdstep = 0.1

channel_names = ['Memb', 'DNA'] + dp.label_names.tolist()

im_paths = list()

classes = torch.Tensor(nclasses, nclasses).fill_(-25).cuda(gpu_id)
for i in range(0,nclasses): classes[i,i] = 0
classes = Variable(classes)

struct = Variable(torch.Tensor(nclasses, nlatent).fill_(0).cuda(gpu_id))

for i in tqdm(range(0, nframes)):
    
    save_path_step = '{0}/step_{1}.png'.format(save_dir, int(i))
    save_path_step_wide = '{0}/step_{1}_wide.png'.format(save_dir, int(i))
    save_path_step_wide_v2 = '{0}/step_{1}_wide_v2.png'.format(save_dir, int(i))    
    save_path_tif = '{0}/step_{1}.ome.tif'.format(save_dir, int(i))
    
    im_paths.append([save_path_step, save_path_step_wide, save_path_step_wide_v2, save_path_tif])
    
    if os.path.exists(save_path_tif) and not args.overwrite:
        continue
    
    ref = Variable(torch.from_numpy(positions_final[i]).float().repeat(nclasses,1).cuda(gpu_id))
    
#     struct.normal_(mean=0, std=1)
    # ref += Variable(torch.Tensor(1, nref).normal_(0,stdstep).repeat(nclasses,1).cuda(gpu_id))
    
    im_out = dec([classes, ref, struct])

    im_out_flat = tensor2img(im_out.data.cpu())
    scipy.misc.imsave(save_path_step, im_out_flat)
    
    im_out_flat = tensor2img(im_out.data.cpu(), do_half=False)
    scipy.misc.imsave(save_path_step_wide, im_out_flat)
    
    im_out_flat = tensor2img(im_out.data.cpu(), proj_all=False, do_half=False)
    scipy.misc.imsave(save_path_step_wide_v2, im_out_flat)
    
    im_out = im_out.data.cpu().numpy()
    
    im_out = np.concatenate([im_out[0,[0]], im_out[0,[2]], im_out[:,1]], axis=0)
    im_out = np.transpose(im_out, (3, 0, 1, 2))
    
    with omeTifWriter.OmeTifWriter(save_path_tif, overwrite_file=True) as w:
        w.save(im_out, channel_names=channel_names)
    
df_impaths = pd.DataFrame(im_paths, columns=['save path', 'save path wide', 'save path wide v2', 'save path tif'])


df_out = pd.concat([df_states, df_impaths], axis=1)
df_out.to_csv(save_dir + os.sep + 'data.csv')
    

    
import scipy.misc as misc    

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

nrows = len(df_out['save path wide v2'])

im_out = list()    

for i in range(0, nrows):
    im = misc.imread(df_out['save path wide v2'][i])
    
#     pdb.set_trace()
    
    im = Image.open(df_out['save path wide v2'][i], "r").convert('RGB')
    
    
    draw = ImageDraw.Draw(im,'RGB')
    font = PIL.ImageFont.load_default()
    draw.text((10,10), str(int(i)), (255, 255, 255))#,font=font)
    
    
    im_out.append(np.asarray(im))
    
im_out_sub1 = np.vstack(im_out[0:int(nrows/2)]);
im_out_sub2 = np.vstack(im_out[int(nrows/2):]);
spacer = np.ones([im_out_sub2.shape[0], 3, 3])*255

im_out = np.hstack([im_out_sub1, spacer, im_out_sub2])

scipy.misc.imsave('{0}/all_steps.png'.format(save_dir), im_out)

im_out = list()    



for i in range(0, nrows,2):
    im = misc.imread(df_out['save path wide v2'][i])
    
#     pdb.set_trace()
    
    im = Image.open(df_out['save path wide v2'][i], "r").convert('RGB')
    
    
#     draw = ImageDraw.Draw(im,'RGB')
#     font = PIL.ImageFont.load_default()
#     draw.text((10,10), str(int(i)), (255, 255, 255))#,font=font)
    
    
    im_out.append(np.asarray(im))
    
im_out = np.vstack(im_out);

scipy.misc.imsave('{0}/all_steps_short.png'.format(save_dir), im_out)























