import pandas as pd
import matplotlib as mpl
import os
import pickle
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt

import pdb

import model_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
args = parser.parse_args()



# parent_dir = './test_aaegan/aaegan3Dv8_v2/'
parent_dir = args.parent_dir
analysis_dir = parent_dir + os.sep + 'analysis'

data_var_dir = analysis_dir + os.sep + 'model_structure_variation_per_cell'

figure_dir = parent_dir + os.sep + 'figures'

#get the mitosis labels

model_dir = parent_dir + os.sep + 'struct_model'
opt = pickle.load(open( '{0}/opt.pkl'.format(model_dir), "rb" ))

mito_file = args.parent_dir + os.sep + 'data_jobs_out_mitotic_annotations.csv'
df_mito = pd.read_csv(mito_file)
df_mito = df_mito[['inputFolder', 'inputFilename', 'outputThisCellIndex', 'MitosisLabel']]

dp = model_utils.load_data_provider(opt.data_save_path, opt.imdir, opt.dataProvider)
df_data = dp.csv_data

df_data = df_data.merge(df_mito, on=['inputFolder', 'inputFilename', 'outputThisCellIndex'], how='left')
df_data = df_data.rename(columns = {'MitosisLabel_y': 'MitosisLabel'})

df_data_labeled = df_data[~np.isnan(df_data['MitosisLabel'])]
labels = df_data_labeled['MitosisLabel']


def print_hist(data, bins, ulabels):
    for i in range(0, len(ulabels)):
            label = ulabels[i]
            data_tmp = data[data['label'] == label]
            
            plt.hist(np.asarray(data_tmp.log_det_scaled), bins=bins, normed=True, alpha = 0.5, label=label, color=colors[int(i)])
                   
    plt.axis(axis)    
    plt.xlabel('variation')
    
def set_axis_style(ax, labels):
#     ax.get_yaxis().set_tick_params(direction='out')
#     ax.yaxis.set_ticks_position('bottom')
    ax.set_yticks(np.arange(1, len(labels) + 1))
    plt.gca().set_yticklabels([])
    ax.set_ylim(0.25, len(labels) + 0.75)
    ax.invert_yaxis()
#     for tick in ax.get_xticklabels():
#         tick.set_rotation(90)    
    
def print_violin(data, bins, ulabels):
    
#     data_list = list()
    for i in range(0, len(ulabels)):
            label = ulabels[i]
            data_tmp = data[data['label'] == label]
            
#             data_list.append(data_tmp.log_det_scaled)
            
#             if i == len(ulabels)-1:
#                 pdb.set_trace()
            
            if len(data_tmp) > 0:
                try:
                    plt.violinplot(np.asarray(data_tmp.log_det_scaled), [i+1], vert=False, showmedians=True, showextrema=False)
                except:
                    pdb.set_trace()
            else:
                #do this to get the plot colors all the same
                plt.violinplot(np.asarray([-1000]), [i+1], vert=False, showmedians=True, showextrema=False)
                   
    plt.axis(axis)    
    set_axis_style(plt.gca(), ulabels)
    plt.xlabel('-log determinant')    
    
    
  
    
################
# DATA VARIATION
################
data = pd.read_csv(data_var_dir + os.sep + 'all_dat.csv')

data['log_det_scaled'] = -data['log_det_scaled']

[ulabels, label_inds] = np.unique(data.label, return_inverse=True)

ulabels_print = ulabels.copy()

ulabels_print[ulabels_print == 'Desmoplakin'] = 'desmoplakin'
ulabels_print[ulabels_print == 'Fibrillarin'] = 'fibrillarin'
ulabels_print[ulabels_print == 'Lamin B1'] = 'lamin B1'

minbin = np.min(data.log_det_scaled)
maxbin = np.max(data.log_det_scaled)

bins = np.linspace(minbin, maxbin, 50)
# axis = [0, len(ulabels), minbin-.25, maxbin+.25]
axis = [minbin-.25, maxbin+.25, 0, len(ulabels)]
figsize = (6,6)
colormap = 'Vega20'



colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(ulabels)+1))*0.8


train_test_dict = {'train': 1, 'test': 2}

for train_or_test in train_test_dict:
    plt.figure(figsize=figsize)
    
    data_tmp = data[data['train_or_test'] == train_or_test]
    
    mito_label = df_data['MitosisLabel'][data_tmp['img_index']]
    mito_inds = np.any(np.vstack([mito_label == target_label for target_label in [0]]), axis=0)
    
    data_tmp = data_tmp[mito_inds]
    
    
    print_violin(data_tmp, bins, ulabels)
    plt.title(train_or_test)
    
    if train_or_test == 'train':
        plt.gca().set_yticklabels(ulabels_print)
    
    plt.savefig('{0}/variation_per_cell_{1}.png'.format(figure_dir, train_or_test), bbox_inches='tight')
    plt.close('all')    
    
train_test_dict = {'train': 1, 'test': 2}

for train_or_test in train_test_dict:
    plt.figure(figsize=figsize)
    
    data_tmp = data[data['train_or_test'] == train_or_test]
    
    mito_label = df_data['MitosisLabel'][data_tmp['img_index']]
    mito_inds = np.any(np.vstack([mito_label == target_label for target_label in [1, 2, 3, 4, 5, 6, 7]]), axis=0)
    
    data_tmp = data_tmp[mito_inds]
    
    print_violin(data_tmp, bins, ulabels)
    plt.title(train_or_test)
    
    if train_or_test == 'train':
        plt.gca().set_yticklabels(ulabels_print)
    
    plt.savefig('{0}/variation_per_cell_mito_{1}.png'.format(figure_dir, train_or_test), bbox_inches='tight')
    plt.close('all')     
    
    