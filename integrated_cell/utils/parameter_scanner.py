import json
import math
import argparse
import sys
import numpy as np

import subprocess
from subprocess import Popen, PIPE

import pdb
import datetime

import logging

import os


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    
    (out, err) = process.communicate()
    
    return out, err
    
def sample_params(param_kvs):
    sampled_param = dict()
    
    for k, v in param_kvs.items():
        sampled_param[k] = sample_param(**v)

        
    return sampled_param

def sample_param(distribution, params, base = 10):
    #currently only 'uniform', and 'log-uniform'
    
    if distribution == 'uniform':
        sample = np.random.uniform(params[0], params[1])
        
    elif distribution == 'log-uniform':
        params = np.log(params)/np.log(base)
        sample = np.power(base, np.random.uniform(params[0], params[1]))
        
    elif distribution == 'sample':
        sample = params[np.random.randint(len(params))]
    
    return sample

def get_logger(save_dir):
    logging.basicConfig(filename=os.path.join(save_dir, 'run.log'),
    format='%(asctime)s %(message)s',
    filemode='w', level=logging.DEBUG)
    logger = logging.getLogger()

    return logger
    
def get_save_dir(save_parent):
    the_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")    
    save_dir = os.path.join(save_parent, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    
    
    return save_dir
    
def parameter_scanner(template, param_kvs, save_parent, verbose = True, niters = int(1E10)):
    """
    A function that samples parameters from distributions as defined in param_kvs
    and runs a command with those parameters.
    
    Parameters
    ----------
    template : string template to map param_ksvs to
    param_kvs: dictionary of distributions to sample from
    niters: the number of times to loop
    log_file: file object to write output of commands to
    
    Returns
    -------
        N/A
    """
    
    for i in range(niters): 
        save_dir = get_save_dir(save_parent)
        
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        logger = get_logger(save_dir)
        param_kvs['save_dir']['params'] = [save_dir]

        
        sampled_params = sample_params(param_kvs)
        
        execution_string = template.format_map(sampled_params)

        if logger: logger.info(execution_string)
        if verbose: print(execution_string)
            
        stdout, stderr = run_command(execution_string)
    
        if logger: 
            logger.debug(stdout)
            logger.error(stderr)
        
        if verbose: print(stderr)
        
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, help='gpu id')
    
    args = parser.parse_args()

    template = 'python /root/projects/pytorch_integrated_cell/train_model.py ' \
                    '--gpu_ids {gpu_ids} ' \
                    '--save_dir {save_dir} ' \
                    '--lrEnc {lrAA} --lrDec {lrAA} ' \
                    '--lrEncD {lrEncD} --lrDecD {lrDecD} ' \
                    '--lambdaEncD {lambdaEncD} --lambdaDecD {lambdaDecD} ' \
                    '--model_name {model_name} ' \
                    '--train_module aaegan_trainv7 ' \
                    '--kwargs_encD \'{{"noise_std": {kwargs_encD_noise}}}\' ' \
                    '--kwargs_decD \'{{"noise_std": {kwargs_decD_noise}}}\' ' \
                    '--kwargs_optim \'{{"betas": [{betas_1}, {betas_2}]}}\' ' \
                    '--imdir /root/results/ipp/ipp_17_10_25 ' \
                    '--dataProvider DataProvider3Dh5 ' \
                    '--saveStateIter 1 --saveProgressIter 1 ' \
                    '--channels_pt1 0 2 --channels_pt2 0 1 2 ' \
                    '--batch_size 16  ' \
                    '--nlatentdim 128 ' \
                    '--nepochs 15 ' \
                    '--nepochs_pt2 0 '
    

    param_kvs = {'gpu_ids': {'distribution': 'sample', 'params':[args.gpu_id]},
            "lrAA": {'distribution': 'log-uniform', 'params':[1E-5, 1E-3]},
            "lrEncD": {'distribution': 'log-uniform', 'params':[1E-5, 1E-3]},
            "lrDecD": {'distribution': 'log-uniform', 'params':[1E-5, 1E-3]},
            "lambdaEncD": {'distribution': 'log-uniform', 'params':[1E-6, 1]},
            "lambdaDecD": {'distribution': 'log-uniform', 'params':[1E-5, 1]},
            "model_name": {'distribution': 'sample', 'params':['aaegan3Dv6-relu-exp', 'aaegan3Dv6-sn']},
            "kwargs_encD_noise": {'distribution': 'sample', 'params':[0]},
            "kwargs_decD_noise": {'distribution': 'uniform', 'params':[0, 0.5]},
            "betas_1": {'distribution': 'uniform', 'params':[0, 0.9]},
            "betas_2": {'distribution': 'uniform', 'params':[0.9, 0.999]},
            "save_dir": {'distribution': 'sample', 'params':['']},}

    save_parent = '/root/results/integrated_cell/param_search'
    

    parameter_scanner(template, param_kvs, save_parent)
    
