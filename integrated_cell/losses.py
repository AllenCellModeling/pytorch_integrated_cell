import torch.nn as nn

from . import utils

import torch

class ClassMSELoss(nn.Module):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`.
    
    Same as nn.MSELoss but the 'target' is an integer that gets converted to one-hot.
    """
    def __init__(self, **kwargs):
        super(ClassMSELoss,self).__init__()
        
        self.mseloss = nn.MSELoss(**kwargs)
        
    def forward(self, input, target):
        
        if input.shape[1] > 1:
            target_onehot = utils.index_to_onehot(target, input.shape[1])
        else: 
            target_onehot = target
        
        return self.mseloss(input, target_onehot)
    


