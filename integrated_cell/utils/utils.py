import torch
import datetime
import os
import json


def index_to_onehot(index, n_classes):
    index = index.long().unsqueeze(1)
    
    onehot = torch.zeros(len(index), n_classes).type_as(index).float()
    onehot.scatter_(1, index, 1)
    
    return onehot
    
def memReport():
    import gc
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    
def cpuStats():
    import psutil
    import sys

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def save_load_dict(save_path, args = None, overwrite = False, verbose = True):
    #saves a dictionary, 'args', as a json file. Or loads if it exists.
    
    if os.path.exists(save_path) and not overwrite:
        warnings.warn(
            "args file exists and overwrite is not set to True. Using existing args file."
        )

        # load argsions file
        with open(save_path, "rb") as f:
            args = json.load(f)
    else:
        # make a copy if the args file exists
        if os.path.exists(save_path):
            the_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            shutil.copyfile(save_path, "{0}_{1}".format(save_path, the_time))

        with open(save_path, "w") as f:
            json.dump(args, f)

    return args   


