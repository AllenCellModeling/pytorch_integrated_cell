import os
import numpy as np
import subprocess
import time
import tqdm
import itertools
import pandas as pd

import pdb

parent_dir = '/root/results/integrated_cell/param_search'
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)
    
scripts_dir = parent_dir + os.sep + 'scripts'
if not os.path.exists(scripts_dir):
    os.makedirs(scripts_dir)


job_list_path = parent_dir + 'job_list.csv'
    
job_template = 'cd .. \n' \
            'python /root/projects/pytorch_integrated_cell/train_model.py \\\n' \
            '\t--gpu_ids {0} \\\n' \
            '\t--save_dir {1} \\\n' \
            '\t--model_name aaegan3Dv6-exp-half \\\n' \
            '\t--lrEnc {2} --lrDec {2} --lrEncD {3} --lrDecD {2} \\\n' \
            '\t--encDRatio {4} --decDRatio {5} \\\n' \
            '\t--noise {6} \\\n' \
            '\t--nlatentdim 128 \\\n' \
            '\t--batch_size 30 \\\n' \
            '\t--nepochs 25 --nepochs_pt2 0 \\\n' \
            '\t--train_module aaegan_trainv6 \\\n' \
            '\t--imdir /root/data/ipp/ipp_17_10_25 \\\n' \
            '\t--dataProvider DataProvider3Dh5-half \\\n' \
            '\t--saveStateIter 1 --saveProgressIter 1 \\\n' \
            '\t--channels_pt1 0 2 --channels_pt2 0 1 2 \\\n'

def format_job(string, gpu_id, row):
    return string.format(gpu_id, row['job_dir'], row['lr'], row['lrDec'], row['encDRatio'], row['decDRatio'], row['noise'])


# job_template = 'sleep {0!s}'
# def format_job(string):
#     return string.format(np.random.choice(10, 1)[0])
job_path_template = scripts_dir + os.sep + 'param_job_{0!s}.sh'
job_dir_template = parent_dir + os.sep + 'param_job_{0!s}'

if os.path.exists(job_list_path):
    job_list = pd.read_csv(job_list_path)
else:
    lr_range = [5E-4, 2E-4, 1E-4, 5E-5, 1E-5, 5E-6]
    lrDec_range = [1E-1, 5E-2, 1E-2, 5E-3, 1E-3, 1E-4, 5E-5, 1E-5]
    encDRatio_range  = [1E-1, 5E-2, 1E-2, 5E-3, 1E-3, 5E-4, 1E-4, 5E-5, 1E-5, 5E-6, 1E-6]
    decDRatio_range = encDRatio_range
    noise_range = [0.1, 0.01, 0.001]

    param_opts = itertools.product(lr_range, lrDec_range, encDRatio_range, decDRatio_range, noise_range)

    opts_list = [list(i) for i in param_opts]
    
    job_scripts = [job_path_template.format(i) for i in range(0, len(opts_list))]
    job_dirs = [job_dir_template.format(i) for i in range(0, len(opts_list))]
    
    all_opts = [opt+[job_dir]+[job_scripts] for opt, job_dir, job_scripts in zip(opts_list, job_dirs, job_scripts)]
    
    job_list = pd.DataFrame(all_opts, columns = ['lr', 'lrDec', 'encDRatio', 'decDRatio', 'noise', 'job_dir', 'job_script'])
    
    job_list.to_csv(job_list_path)

    
# randomly shuffle
job_list = job_list.sample(frac=1)




processes = range(0, 8)
job_process_list = [-1] * len(processes)

#for every job
for index, row in job_list.iterrows():
    
    #check to see if the job exists
    job_script = row['job_script']
    if os.path.exists(job_script):
        continue

    #if it doesnt exist, wait for 
    job_status = None

    #look for a process that isn't running
    while job_status is None:
        for job_process, process_id in zip(job_process_list, processes):
            
            #check process status
            if job_process == -1:
                job_status = True
                break
            else:
                job_status = job_process.poll()
                
                if job_status is not None:
                    break
                    
        if job_status is not None:
            break

        print('Waiting for next job')
        time.sleep(3)

    #start a job
    #get the job string
    job_str = format_job(job_template, process_id, row)

    #write the bash file to disk

    print('starting ' + job_script + ' on process ' + str(process_id))
    with open(job_script, 'w') as text_file:
        text_file.write(job_str)

    pdb.set_trace()
    job_process_list[process_id] = subprocess.Popen('bash ' + job_script, stdout=subprocess.PIPE, shell=True)

print(job_template)         