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


#####
# Figure settings
#####

linewidth = 1    
figsize = (7.5, 1.5)
colormap = 'Vega20'  

color_offset = 2

cmap = np.array(mpl.cm.get_cmap(colormap).colors)
    
# cmap = cmap[np.round(np.linspace(0, len(cmap)-1, 5)).astype(int)]
    
group_spacing = 0.65    
fontsize = 7


plt.rc('font', **{'family': 'serif', 'serif':['DejaVu Serif'], 'size': fontsize})
plt.rc('lines', **{'linewidth': linewidth, 'markeredgewidth': linewidth})

group_names =['gen train', 'gen test']



    
def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(0, len(labels)*4, 4)+1.25)
    plt.gca().set_xticklabels([])
    ax.set_xlim(-1.25, len(labels)*4)
#     ax.invert_xaxis()
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)    
    
def print_violin(data, ulabels, position_modifier=0, cmap = [0,0,0], label = None):
    
#     data_list = list()
    for i in range(0, len(ulabels)):
            label = ulabels[i]
            data_tmp = data[data['label'] == label]
            
            
            if len(data_tmp) > 0:
                try:
                    violin_parts = plt.violinplot(np.asarray(data_tmp.log_det_scaled), 
                                                  [i*4 + position_modifier], vert=True, showmedians=True, showextrema=False)
                except:
                    pdb.set_trace()
            else:
                #do this to get the plot colors all the same
                violin_parts = plt.violinplot(np.asarray([-1000]), 
                                              [i*4 + position_modifier], vert=True, showmedians=True, showextrema=False)
                   
            for pc in violin_parts['bodies']:
#                 pc.set_color(cmap)
                pc.set_facecolor(cmap)
#                 pc.set_edgecolor(cmap)
            
            violin_parts['cmedians'].set_edgecolor(cmap)

    h = mpl.patches.Patch(color=cmap, label=label)
    plt.axis(axis)    
    set_axis_style(plt.gca(), ulabels)
    plt.ylabel('-log determinant')    
    
    return h
  

#####
# Data Setup
#####

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
    
    
handles = list()   
c = 0      
    
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
ulabels_print[ulabels_print == 'Alpha actinin'] = r'$\alpha$-actinin'
ulabels_print[ulabels_print == 'Alpha tubulin'] = r'$\alpha$-tubulin'
ulabels_print[ulabels_print == 'Beta actin'] = r'$\beta$-actin'
ulabels_print[ulabels_print == 'Sec61 beta'] = r'Sec61 $\beta$'

minbin = np.min(data.log_det_scaled)
maxbin = np.max(data.log_det_scaled)

bins = np.linspace(minbin, maxbin, 50)
# axis = [0, len(ulabels), minbin-.25, maxbin+.25]
axis = [0, len(ulabels), minbin-.25, maxbin+.25]

colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(ulabels)+1))*0.8


train_test_dict = {'train': 1, 'test': 2}


plt.figure(figsize=figsize)

c2 = 0

for train_or_test in train_test_dict:
    
    
    data_tmp = data[data['train_or_test'] == train_or_test]
    
    h = print_violin(data_tmp, ulabels, position_modifier = c*group_spacing+1, cmap = cmap[c+c2+color_offset], label = group_names[c])
    
    handles.append(h)
    c+=1
    c2 +=1


leg = plt.legend(handles, group_names, fontsize = fontsize,
                    loc=2,
                    borderaxespad=0,
                    frameon=False
                )


ax = plt.gca()
# pdb.set_trace()

prefix = '     '

#this is to make the horizontal spacing the same as with the other print variation figures script
ytick = [prefix + str(int(item)) if item == int(item) else prefix + str(item) for item in ax.get_yticks()]
ax.set_yticklabels(ytick)

leg.get_frame().set_linewidth(0.0)

plt.gca().set_xticklabels(ulabels_print)    
    
plt.savefig('{0}/variation_per_cell_v2_no_mito.png'.format(figure_dir, train_or_test),  dpi=300, bbox_inches='tight')
plt.close('all')     
    
    