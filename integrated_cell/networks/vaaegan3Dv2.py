from torch import nn
import torch
import pdb
from integrated_cell.model_utils import init_opts
import numpy as np
import integrated_cell.utils.spectral_norm as spectral_norm
import integrated_cell.models.bvae as bvae

ksize = 4
dstep = 2

class Enc(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, nch, gpu_ids):
        super(Enc, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = 2
        
        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef
        
        self.first = nn.Sequential(
                            nn.Conv3d(nch, 64, ksize, dstep, 1),
                            nn.BatchNorm3d(64),
                            nn.ReLU(inplace=True))
        
        layer_sizes = (2**np.arange(6, 12))
        layer_sizes[layer_sizes>1024] = 1024
        
        self.poolBlocks = nn.ModuleList([])
        self.transitionBlocks = nn.ModuleList([])
        
        for i in range(1, len(layer_sizes)):
            self.poolBlocks.append(nn.AvgPool3d(2**i, stride=2**i))
            self.transitionBlocks.append(nn.Sequential(
                                            spectral_norm(nn.Conv3d(int(layer_sizes[i-1]+nch), int(layer_sizes[i]), ksize, dstep, 1)),
                                            nn.BatchNorm3d(int(layer_sizes[i])),
                                            nn.ReLU(inplace=True),
                                        ))

        
        if self.nClasses > 0:
            self.classOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nClasses),
                # nn.BatchNorm1d(self.nClasses),
                nn.LogSoftmax()
            )
        
        if self.nRef > 0:
            self.refOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nRef),
                # nn.BatchNorm1d(self.nRef)
            )
        
        if self.nLatentDim > 0:
            self.latentOutMu = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nLatentDim))

            self.latentOutLogSigma = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nLatentDim))
            
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
            
        x_tmp = nn.parallel.data_parallel(self.first, x, gpu_ids)
        
        for pool, trans in zip(self.poolBlocks, self.transitionBlocks):
            x_sub = pool(x)
            x_tmp = nn.parallel.data_parallel(trans, torch.cat([x_sub, x_tmp], 1), gpu_ids)
        
        
        x = x_tmp.view(x_tmp.size()[0], 1024*int(self.fcsize*1*1))
        
        xOut = list()
                
        if self.nClasses > 0:
            xClasses = nn.parallel.data_parallel(self.classOut, x, gpu_ids)
            xOut.append(xClasses)
            
        if self.nRef > 0: 
            xRef = nn.parallel.data_parallel(self.refOut, x, gpu_ids)
            xOut.append(xRef)
        
        if self.nLatentDim > 0:
            xLatentMu = nn.parallel.data_parallel(self.latentOutMu, x, gpu_ids)
            xLatentLogSigma = nn.parallel.data_parallel(self.latentOutLogSigma, x, gpu_ids)
            
            if self.training:
                xOut.append([xLatentMu, xLatentLogSigma])
            else:
                xOut.append(bvae.reparameterize(xLatentMu, xLatentLogSigma, add_noise=False))
        
        return xOut
    
class Dec(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, nch, gpu_ids, output_padding = (0,1,0)):
        super(Dec, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = 2
        
        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef
        
        self.fc = nn.Linear(self.nLatentDim + self.nClasses + self.nRef, 1024*int(self.fcsize*1*1))
        
        self.main = nn.Sequential(
            nn.BatchNorm3d(1024),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(1024, 1024, ksize, dstep, 1, output_padding = output_padding),
            nn.BatchNorm3d(1024),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(1024, 512, ksize, dstep, 1),
            nn.BatchNorm3d(512),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(512, 256, ksize, dstep, 1),
            nn.BatchNorm3d(256),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, ksize, dstep, 1),
            nn.BatchNorm3d(128),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, ksize, dstep, 1),
            nn.BatchNorm3d(64),
            
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, nch, ksize, dstep, 1),
            # nn.BatchNorm3d(nch),
            nn.Sigmoid()             
        )
            
    def forward(self, xIn):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        if self.nClasses > 0:
            xIn[0] = torch.exp(xIn[0])
        
        x = torch.cat(xIn, 1)
        
        x = self.fc(x)
        x = x.view(x.size()[0], 1024, self.fcsize, 1, 1)
        
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
        
        self.start = spectral_norm(nn.Conv3d(nch, 64, ksize, dstep, 1))
        
        
        layer_sizes = (2**np.arange(6, 12))
        layer_sizes[layer_sizes>1024] = 1024
        
        self.poolBlocks = nn.ModuleList([])
        self.transitionBlocks = nn.ModuleList([])
        
        for i in range(1, len(layer_sizes)):
            self.poolBlocks.append(nn.AvgPool3d(2**i, stride=2**i))
            self.transitionBlocks.append(nn.Sequential(
                                            nn.LeakyReLU(0.2, inplace=True),
                                            spectral_norm(nn.Conv3d(int(layer_sizes[i-1]+nch), int(layer_sizes[i]), ksize, dstep, 1)),
                                            nn.BatchNorm3d(int(layer_sizes[i])),
                                        ))
    
        self.end = nn.LeakyReLU(0.2, inplace=True)
        self.fc = nn.Linear(1024*int(self.fcsize), nout)


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
        
        x_tmp = nn.parallel.data_parallel(self.start, x, gpu_ids)
        
        for pool, trans in zip(self.poolBlocks, self.transitionBlocks):
            x_sub = pool(x)
            x_tmp = nn.parallel.data_parallel(trans, torch.cat([x_sub, x_tmp], 1), gpu_ids)
        
        x = self.end(x_tmp)
        x = x.view(x.size()[0], 1024*int(self.fcsize))
        x = self.fc(x)
        
        return x
    



