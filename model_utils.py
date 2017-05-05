import torch

def set_gpu_recursive(var, gpu_id):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = set_gpu_recursive(var[key], gpu_id)
        else:
            try:
                if gpu_id != -1:
                    var[key] = var[key].cuda(gpu_id)
                else:
                    var[key] = var[key].cpu()
            except:
                pass
    return var  

def sampleUniform (batsize, nlatentdim): 
    return torch.Tensor(batsize, nlatentdim).uniform_(-1, 1)

def sampleGaussian (batsize, nlatentdim): 
    return torch.Tensor(batsize, nlatentdim).normal_()