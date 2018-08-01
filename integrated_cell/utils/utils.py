import torch


def index_to_onehot(index, n_classes):
    index = index.long().unsqueeze(1)
    
    onehot = torch.zeros(len(index), n_classes).type_as(index).float()
    onehot.scatter_(1, index, 1)
    
    return onehot
    
