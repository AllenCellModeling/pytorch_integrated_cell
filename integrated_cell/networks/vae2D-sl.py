from torch import nn
import torch
import pdb
from integrated_cell.model_utils import init_opts

import integrated_cell.utils.spectral_norm as spectral_norm

import integrated_cell.models.bvae as bvae

ksize = 4
dstep = 2

def get_activation(activation):
    if activation.lower() == 'relu':
        return nn.ReLU()
    if activation.lower() == 'prelu':
        return nn.PReLU()

class Basic2DLayer(nn.Module):
    def __init__(self, ch_in, ch_out, ksize, dstep, activation):
        super(Basic2DLayer, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, ksize, dstep, 1),
            nn.BatchNorm2d(ch_out),
            get_activation(activation)
        )
        
    def forward(self, x):
        return self.main(x)
    

class Enc(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, nch, gpu_ids, activation='ReLU'):
        super(Enc, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef

        self.main = nn.Sequential(
            Basic2DLayer(nch, 64, ksize, dstep, activation),
            Basic2DLayer(64, 128, ksize, dstep, activation),
            Basic2DLayer(128, 256, ksize, dstep, activation),
            Basic2DLayer(256, 512, ksize, dstep, activation),
            Basic2DLayer(512, 1024, ksize, dstep, activation),
            Basic2DLayer(1024, 1024, ksize, dstep, activation)
        )

        if self.nClasses > 0:
            self.classOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nClasses),
                # nn.BatchNorm1d(self.nClasses),
                nn.LogSoftmax()
            )

        if self.nRef > 0:
            self.refOutMu = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nRef))

        if self.nLatentDim > 0:
            self.latentOutMu = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nLatentDim))

            self.latentOutLogSigma = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nLatentDim))

    def forward(self, x, reparameterize=False):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        activations = list()
        for layer in self.main:
            x = layer(x)
            activations.append(x)
        
        x = x.view(x.size()[0], 1024*int(self.fcsize*1*1))

        xOut = list()
        
        if self.nClasses > 0:
            xClasses = nn.parallel.data_parallel(self.classOut, x, gpu_ids)
            xOut.append(xClasses)

        if self.nRef > 0:
            xRefMu = nn.parallel.data_parallel(self.refOutMu, x, gpu_ids)
            xOut.append(xRefMu)

        if self.nLatentDim > 0:
            xLatentMu = nn.parallel.data_parallel(self.latentOutMu, x, gpu_ids)
            xLatentLogSigma = nn.parallel.data_parallel(self.latentOutLogSigma, x, gpu_ids)
            
            if self.training:
                xOut.append([xLatentMu, xLatentLogSigma])
                
                return xOut, activations
            else:
                xOut.append(bvae.reparameterize(xLatentMu, xLatentLogSigma, add_noise=False))
                
                return xOut

class Dec(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, nch, gpu_ids, activation='ReLU', exponentiate_classes = True, fcsize = 2, output_padding = (0,1,0)):
        super(Dec, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = fcsize

        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef
        
        self.output_padding = output_padding
        
        self.exponentiate_classes = exponentiate_classes

        self.fc = nn.Linear(self.nLatentDim + self.nClasses + self.nRef, 1024*int(self.fcsize*1*1))

        self.main = nn.Sequential(
            nn.BatchNorm2d(1024),

            get_activation(activation),
            nn.ConvTranspose2d(1024, 1024, ksize, dstep, 1, output_padding = output_padding),
            nn.BatchNorm2d(1024),

            get_activation(activation),
            nn.ConvTranspose2d(1024, 512, ksize, dstep, 1),
            nn.BatchNorm2d(512),

            get_activation(activation),
            nn.ConvTranspose2d(512, 256, ksize, dstep, 1),
            nn.BatchNorm2d(256),

            get_activation(activation),
            nn.ConvTranspose2d(256, 128, ksize, dstep, 1),
            nn.BatchNorm2d(128),

            get_activation(activation),
            nn.ConvTranspose2d(128, 64, ksize, dstep, 1),
            nn.BatchNorm2d(64),

            get_activation(activation),
            nn.ConvTranspose2d(64, nch, ksize, dstep, 1),
            # nn.BatchNorm2d(nch),
            nn.Sigmoid()
        )

    def forward(self, xIn):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        if self.nClasses > 0 and self.exponentiate_classes:
            xIn[0] = torch.exp(xIn[0])

        x = torch.cat(xIn, 1)

        x = self.fc(x)
        x = x.view(x.size()[0], 1024, self.fcsize, 1,)

        x = nn.parallel.data_parallel(self.main, x, gpu_ids)

        return x

