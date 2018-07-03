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


#####
# Figure settings
#####

linewidth = 1    

figsize = (7.5, 1.5)
colormap = 'Vega10'  
cmap = mpl.cm.get_cmap(colormap).colors
    
group_spacing = 0.65    
fontsize = 7


plt.rc('font', **{'family': 'serif', 'serif':['DejaVu Serif'], 'size': fontsize})
plt.rc('lines', **{'linewidth': linewidth, 'markeredgewidth': linewidth})


labels = ['data', 'AE - train', 'AE - test', 'gen']


# parent_dir = './test_aaegan/aaegan3Dv8_v2/'
parent_dir = args.parent_dir
analysis_dir = parent_dir + os.sep + 'analysis'

data_var_dir = analysis_dir + os.sep + 'data_variation'

figure_dir = parent_dir + os.sep + 'figures'


    
def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(0, len(labels)*4, 4)+1.25)
    plt.gca().set_xticklabels([])
    ax.set_xlim(-1.25, len(labels)*4)
#     ax.invert_xaxis()
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)    
    
def print_violin(data_var_info, ulabels, position_modifier=0, cmap = [0,0,0], label = None):
    
    data_list = list()
    for row in tqdm(data_var_info.iterrows()):
        with open(row[1]['save_path'], 'rb') as file:
            data = pickle.load(file)
            
            try:
                triu_vals = data['corr_mat_struct'][np.triu_indices(data['corr_mat_struct'].shape[0], k=1)]
            except:
                triu_vals = data['corr_mat'][np.triu_indices(data['corr_mat'].shape[0], k=1)]

#             triu_vals = triu_vals[np.random.rand(triu_vals.shape[0]) <= 0.02]
                
            violin_parts = plt.violinplot(triu_vals, [row[1]['label_id']*4 + position_modifier], vert=True, showmedians=True, showextrema=False)
            
            for pc in violin_parts['bodies']:
#                 pc.set_color(cmap)
                pc.set_facecolor(cmap)
#                 pc.set_edgecolor(cmap)
            
            violin_parts['cmedians'].set_edgecolor(cmap)

            
    h = mpl.patches.Patch(color=cmap, label=label)
    plt.axis(axis)    
    set_axis_style(plt.gca(), ulabels)
    plt.ylabel('correlation')        
    
    return h
    

handles = list()   
c = 0    
################
# DATA VARIATION
################
data_var_dir = analysis_dir + os.sep + 'data_variation'
data_var_info = pd.read_csv(data_var_dir + os.sep + 'info.csv')

[ulabels, label_inds] = np.unique(data_var_info.label_name, return_inverse=True)

ulabels[ulabels == 'Desmoplakin'] = 'desmoplakin'
ulabels[ulabels == 'Fibrillarin'] = 'fibrillarin'
ulabels[ulabels == 'Lamin B1'] = 'lamin B1'
ulabels[ulabels == 'Alpha actinin'] = r'$\alpha$-actinin'
ulabels[ulabels == 'Alpha tubulin'] = r'$\alpha$-tubulin'
ulabels[ulabels == 'Beta actin'] = r'$\beta$-actin'
ulabels[ulabels == 'Sec61 beta'] = r'Sec61 $\beta$'

xrange = [0, (len(ulabels)+2)*4]
yrange = [-0.05, 1]
axis = xrange + yrange

plt.figure(figsize=figsize)
h = print_violin(data_var_info, label_inds, cmap = cmap[c], label = labels[c])

handles.append(h)
c+=1


#################
# Encode-Decode variation
#################
    
data_var_dir = analysis_dir + os.sep + 'model_structure_variation'
data_var_info = pd.read_csv(data_var_dir + os.sep + 'info.csv')

train_test_dict = {'train': 1, 'test': 2}

for train_or_test in train_test_dict:
    
    data_var_info_tmp = data_var_info[data_var_info['train_or_test'] == train_or_test]
    h = print_violin(data_var_info_tmp, label_inds, position_modifier= c * group_spacing, cmap = cmap[c], label = labels[c])
    handles.append(h)

    c+=1
    
    
data_var_dir = analysis_dir + os.sep + 'model_structure_variation_sampled'
data_var_info = pd.read_csv(data_var_dir + os.sep + 'info.csv')

#################
# SAMPLED VARIATION
#################

h = print_violin(data_var_info, label_inds, position_modifier = c * group_spacing, cmap = cmap[c],  label = labels[c])
handles.append(h)

    
# leg = plt.legend(handles, labels, fontsize = fontsize,
#                     bbox_to_anchor=(1.01,0.5), 
#                     loc="center left",
#                     borderaxespad=0,
#                 )

leg = plt.legend(handles, labels, fontsize = fontsize,
                    loc=2,
                    borderaxespad=0,
                    frameon=False
                )

leg.get_frame().set_linewidth(0.0)

ax = plt.gca()
ax.set_xticklabels(ulabels)

plt.savefig('{0}/distr_v2.png'.format(figure_dir), dpi=300, bbox_inches='tight') 


#this is to make the horizontal spacing the same as with the other print variation figures script
xtick = [' ' for item in ax.get_xticklabels()]
ax.set_xticklabels(xtick)
# plt.gca().set_xticklabels(ulabels)

plt.savefig('{0}/distr_v2_no_xtick.png'.format(figure_dir), dpi=300, bbox_inches='tight')
plt.close('all')  




    