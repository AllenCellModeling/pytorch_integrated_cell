from torch import nn
import torch
import pdb
from integrated_cell.model_utils import init_opts
import numpy as np

ksize = 4
dstep = 2


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution w/o padding"""
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

def convTranspose2x2(in_channels, out_channels, stride=1):
    """2x2 transposed convolution with padding"""
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0, bias=False)

class ResidualLayer(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate = 0, activate_last = True):
        super(ResidualLayer, self).__init__()
        
        bn_channels = int(np.ceil(in_channels/4))
        
        self.relu = nn.ReLU()
        
        self.conv1 = conv1x1(in_channels, bn_channels)
        self.bn1 = nn.BatchNorm3d(bn_channels)
        
        self.conv2 = conv3x3(bn_channels, bn_channels, stride)
        self.bn2 = nn.BatchNorm3d(bn_channels)
        
        self.conv3 = conv1x1(bn_channels, out_channels)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.stride = stride
        
        self.downsample = None
        if stride > 1:
            self.downsample = nn.AvgPool3d(stride, stride)
            
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout3d(dropout_rate)

        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
            
        self.activate_last = activate_last
            
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.activate_last:
            out = self.relu(out)
        
        if self.dropout is not None:
            residual = self.dropout(residual)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        if self.projection is not None:
            residual = self.projection(residual)
                    
        out += residual

        return out
    
class ResidualLayerTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2, dropout_rate = 0, activate_last = True):
        super(ResidualLayerTranspose, self).__init__()
        
        bn_channels = int(np.ceil(in_channels/4))
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = conv1x1(in_channels, bn_channels, 1)
        self.bn1 = nn.BatchNorm3d(bn_channels)
        
        self.conv2 = nn.ConvTranspose3d(bn_channels, bn_channels, 4, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(bn_channels)
        
        self.conv3 = conv1x1(bn_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.stride = stride

        self.upsample = None
        if stride > 1:
            self.upsample = nn.Upsample(scale_factor = stride)
            
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout3d(dropout_rate)
        
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv3d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
            
        self.activate_last = activate_last
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.activate_last:
            out = self.relu(out)
        
        if self.dropout is not None:
            residual = self.dropout(residual)
        
        if self.projection is not None:
            residual = self.projection(residual)        
            
        if self.upsample is not None:
            residual = self.upsample(residual)
            
        out += residual


        return out    

class conv3DLookupLayer(nn.Module):
    def __init__(self, n_query_dims, n_ch_in, n_ch_out, kernel_size = 1):
        super(conv3DLookupLayer, self).__init__()
        
        self.n_ch_in = n_ch_in
        self.n_ch_out = n_ch_out
        self.kernel_size = kernel_size
        
        self.lookup_table = torch.nn.Parameter(torch.Tensor(n_query_dims, (kernel_size**3) * n_ch_in*n_ch_out).normal_(0.0, 0.02))
    
    def forward(self, query, target):

        weights = torch.mm(query, self.lookup_table)
        weights = weights.view(query.shape[0], self.n_ch_out, self.n_ch_in, self.kernel_size, self.kernel_size, self.kernel_size)
    
        x_out = [nn.functional.conv3d(x.unsqueeze(0), w) for x, w in zip(target, weights)]
        x_out = torch.cat(x_out,0)
            
        return x_out
    
class multiHeadedConv3DLookupLayer(nn.Module):
    def __init__(self, n_query_dims, n_ch_in, n_ch_out, kernel_size = 1, n_heads = 8):
        super(multiHeadedConv3DLookupLayer, self).__init__()
        
        self.layers = nn.ModuleList([conv3DLookupLayer(n_query_dims, n_ch_in, n_ch_out, kernel_size = 1)
            for i in range(n_heads)])
            
    def forward(self, query, target):
        x_out = [layer(query, target) for layer in self.layers]
        x_out = torch.cat(x_out,1)
        
        return x_out
    
class Enc(nn.Module):
    def __init__(self, n_classes, gpu_ids, n_proj_dims = 512, n_lookup_heads = 6):
        super(Enc, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = 2
        
        self.head_1 = nn.Sequential(
            ResidualLayer(1, 64, 2),
            # ResidualLayer(64, 64, 1),
            ResidualLayer(64, 128, 2),
            # ResidualLayer(128, 128, 1),
            ResidualLayer(128, 256, 2),
            # ResidualLayer(256, 256, 1)
            )
        
        self.lookup_table = multiHeadedConv3DLookupLayer(n_query_dims = n_classes, n_ch_in = 256, n_ch_out = n_proj_dims, kernel_size = 1, n_heads = n_lookup_heads)
        
        self.bn = nn.BatchNorm3d(n_proj_dims*n_lookup_heads)
        self.relu = nn.ReLU(inplace=True)
        
        self.head_2 = nn.Sequential(
            ResidualLayer(n_proj_dims*n_lookup_heads, 1024, 2),
            ResidualLayer(1024, 1024, 2),
            ResidualLayer(1024, 1024, 1),
            ResidualLayer(1024, 1024, 1, activate_last = False),
            )

            
    def forward(self, x, y):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
            
        x_shapes = torch.Tensor([x_tmp.shape[0] for x_tmp in x]).int()
        x_stops = torch.cumsum(x_shapes, 0).int()
        x_starts = x_stops - x_shapes
           
        x = [x_tmp.unsqueeze(1) for x_tmp in x]
            
        x = torch.cat(x, 0)

        
        x = nn.parallel.data_parallel(self.head_1, x, gpu_ids)        
        x = self.lookup_table(y, x)
        
        #add together the images as they appeared in the channel dimension 
        x = [torch.sum(x[start:stop],0).unsqueeze(0) for start, stop in zip(x_starts, x_stops)]
        
        x = torch.cat(x,0)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.head_2(x)
        return x
    
class Dec(nn.Module):
    def __init__(self, n_classes, gpu_ids, n_proj_dims = 128, n_lookup_heads = 8):
        super(Dec, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = 2
        
        self.nLatentDim = 1
        self.nClasses = 1
        self.nRef = 1
        
        nch = 1
        
        self.lookup_table = multiHeadedConv3DLookupLayer(n_query_dims = n_classes, n_ch_in = 1024, n_ch_out = n_proj_dims, kernel_size = 1, n_heads = n_lookup_heads)
        
        self.main = nn.Sequential(
            nn.BatchNorm3d(n_proj_dims*n_lookup_heads),
            
            nn.ReLU(inplace=True),
            
            # ResidualLayerTranspose(n_proj_dims*n_lookup_heads, 512, 2),    
            # # ResidualLayer(512, 512, 1),
            # ResidualLayerTranspose(512, 256, 2),
            # # ResidualLayer(256, 256, 1),
            # ResidualLayerTranspose(256, 128, 2),
            # # ResidualLayer(128, 128, 1),
            # ResidualLayerTranspose(128, 32, 2),
            # # ResidualLayer(64, 64, 1),
            # ResidualLayerTranspose(32, 1, 2),
            # # ResidualLayer(256, 256, 1),
            
            
            nn.ConvTranspose3d(n_proj_dims*n_lookup_heads, 512, ksize, dstep, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            
            # ResidualLayer(512, 512, 1),
            
            nn.ConvTranspose3d(512, 256, ksize, dstep, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # ResidualLayer(256, 256, 1),
            
            nn.ConvTranspose3d(256, 128, ksize, dstep, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # ResidualLayer(128, 128, 1),
            
            nn.ConvTranspose3d(128, 32, ksize, dstep, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # ResidualLayer(64, 64, 1),
            
            nn.ConvTranspose3d(32, nch, ksize, dstep, 1),
            # nn.BatchNorm3d(nch),
            nn.Sigmoid()             
        )
            
    def forward(self, x_in, y_in):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        x = self.lookup_table(y_in, x_in)
        
        # pdb.set_trace()
        
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
 
        return x    
    
class EncD(nn.Module):
    def __init__(self, nlatentdim, nClasses, gpu_ids, **kwargs):
        super(EncD, self).__init__()
        
        self.nfc = 1024
        self.noise_std = 0
        
        for key in ('noise_std', 'nfc'):
            if key in kwargs:
                setattr(self, key, kwargs[key])
          
        self.noise = torch.zeros(0)
        
        self.nClasses = nClasses;
        self.gpu_ids = gpu_ids
        
        nfc = self.nfc
        
        self.main = nn.Sequential(
            nn.Linear(nlatentdim, nfc),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Linear(nfc, nfc),
            nn.BatchNorm1d(nfc),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Linear(nfc, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Linear(512, nClasses)
        )
        

    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        if self.noise_std > 0:
            #allocate an appropriately sized variable if it does not exist
            if self.noise.size() != x.size():
                self.noise = torch.zeros(x.size()).cuda(gpu_ids[0])

            #sample random noise
            self.noise.normal_(mean=0, std=self.noise_std)
            noise = torch.autograd.Variable(self.noise)

            #add to input
            x = x + noise
            
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)

        return x
    
class DecD(nn.Module):
    def __init__(self, nout, nch, gpu_ids, **kwargs):
        super(DecD, self).__init__()
        
        self.noise_std = 0
        for key in ('noise_std'):
            if key in kwargs:
                setattr(self, key, kwargs[key])
          
        self.noise = torch.zeros(0)
        
        self.gpu_ids = gpu_ids
        self.fcsize = 2
        
        self.noise = torch.zeros(0)
        
        self.main = nn.Sequential(
            nn.Conv3d(nch, 64, ksize, dstep, 1),
            # nn.BatchNorm3d(64),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, ksize, dstep, 1),
            nn.BatchNorm3d(128),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, ksize, dstep, 1),
            nn.BatchNorm3d(256),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, ksize, dstep, 1),
            nn.BatchNorm3d(512),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1024, ksize, dstep, 1),
            nn.BatchNorm3d(1024),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1024, 1024, ksize, dstep, 1),
            nn.BatchNorm3d(1024),
            
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Linear(1024*int(self.fcsize), nout)

        # if nout == 1:
        #     self.nlEnd = nn.Sigmoid()
        # else:
        #     self.nlEnd = nn.LogSoftmax()
            
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        if self.noise_std > 0:
            #allocate an appropriately sized variable if it does not exist
            if self.noise.size() != x.size():
                self.noise = torch.zeros(x.size()).cuda(gpu_ids[0])

            #sample random noise
            self.noise.normal_(mean=0, std=self.noise_std)
            noise = torch.autograd.Variable(self.noise)

            #add to input
            x = x + noise
        
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 1024*int(self.fcsize))
        x = self.fc(x)
        
        return x
    



