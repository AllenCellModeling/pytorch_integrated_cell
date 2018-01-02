import pandas as pd
import matplotlib as mpl
import os
import pickle
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt

import pdb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', help='save dir')
args = parser.parse_args()


from tqdm import tqdm

# parent_dir = './test_aaegan/aaegan3Dv8_v2/'
parent_dir = args.parent_dir
analysis_dir = parent_dir + os.sep + 'analysis'

data_var_dir = analysis_dir + os.sep + 'data_variation'

figure_dir = parent_dir + os.sep + 'figures'


def print_hist(data_var_info, bins, label_inds):
    for row in data_var_info.iterrows():
        with open(row[1]['save_path'], 'rb') as file:
            data = pickle.load(file)
            
            try:
                triu_vals = data['corr_mat_struct'][np.triu_indices(data['corr_mat_struct'].shape[0])]
            except:
                triu_vals = data['corr_mat'][np.triu_indices(data['corr_mat'].shape[0])]

            plt.hist(triu_vals, bins=bins, normed=True, alpha = 0.5, label=data['label_name'], 
                     color=colors[np.where(label_inds == row[1]['label_id'])[0]])
    plt.axis(axis)    
    plt.xlabel('correlation')
    
def set_axis_style(ax, labels):
#     ax.get_yaxis().set_tick_params(direction='out')
#     ax.yaxis.set_ticks_position('bottom')

    ax.set_yticks(np.arange(0, len(labels)))
    plt.gca().set_yticklabels([])
    ax.set_ylim(-1.25, len(labels))
    ax.invert_yaxis()
#     for tick in ax.get_xticklabels():
#         tick.set_rotation(90)    
    
def print_violin(data_var_info, bins, ulabels):
    
    data_list = list()
    for row in tqdm(data_var_info.iterrows()):
        with open(row[1]['save_path'], 'rb') as file:
            data = pickle.load(file)
            
            try:
                triu_vals = data['corr_mat_struct'][np.triu_indices(data['corr_mat_struct'].shape[0], k=1)]
            except:
                triu_vals = data['corr_mat'][np.triu_indices(data['corr_mat'].shape[0], k=1)]

            ## comment in for fast testing
#             triu_vals = triu_vals[np.random.rand(triu_vals.shape[0]) <= 0.01]
            
            plt.violinplot(triu_vals, [row[1]['label_id']], vert=False, showmedians=True, showextrema=False)

                   
    plt.axis(axis)    
    set_axis_style(plt.gca(), ulabels)
    plt.xlabel('correlation')        
    
    
    

figsize = (6,6)
colormap = 'Vega20'  
    
################
# DATA VARIATION
################
data_var_dir = analysis_dir + os.sep + 'data_variation'
data_var_info = pd.read_csv(data_var_dir + os.sep + 'info.csv')

[ulabels, label_inds] = np.unique(data_var_info.label_name, return_inverse=True)

ulabels[ulabels == 'Desmoplakin'] = 'desmoplakin'
ulabels[ulabels == 'Fibrillarin'] = 'fibrillarin'
ulabels[ulabels == 'Lamin B1'] = 'lamin B1'

colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(ulabels)+1))*0.8

bins = np.arange(0, 1, 0.005)
axis = [-0.05, 1, 0, len(ulabels)+2]

plt.figure(figsize=figsize)
print_violin(data_var_info, bins, label_inds)

plt.title('data')

plt.gca().set_yticklabels(ulabels)
plt.savefig('{0}/distr.png'.format(figure_dir), bbox_inches='tight')
plt.close('all')   

#################
# Encode-Decode variation
#################
    
data_var_dir = analysis_dir + os.sep + 'model_structure_variation'
data_var_info = pd.read_csv(data_var_dir + os.sep + 'info.csv')

train_test_dict = {'train': 1, 'test': 2}

for train_or_test in train_test_dict:
    plt.figure(figsize=figsize)
    
    data_var_info_tmp = data_var_info[data_var_info['train_or_test'] == train_or_test]
    print_violin(data_var_info_tmp, bins, label_inds)
    plt.title(train_or_test)
    
    if train_or_test == 'train':
        plt.gca().set_yticklabels(ulabels)
    
    plt.savefig('{0}/distr_{1}.png'.format(figure_dir, train_or_test), bbox_inches='tight')
    plt.close('all')    
    
    
data_var_dir = analysis_dir + os.sep + 'model_structure_variation_sampled'
data_var_info = pd.read_csv(data_var_dir + os.sep + 'info.csv')

#################
# SAMPLED VARIATION
#################

# pdb.set_trace()
plt.figure(figsize=figsize)
print_violin(data_var_info, bins, label_inds)
plt.title('sampled')
# plt.legend( loc=1, borderaxespad=0, frameon=False)
plt.savefig('{0}/distr_sampled.png'.format(figure_dir), bbox_inches='tight')
plt.close('all')      
    