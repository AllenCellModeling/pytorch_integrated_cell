from torch import nn
import torch
import pdb
from integrated_cell.model_utils import init_opts

import integrated_cell.utils.spectral_norm as spectral_norm

ksize = 4
dstep = 2

class Enc(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, nch, gpu_ids, opt=None):
        super(Enc, self).__init__()

        self.gpu_ids = gpu_ids
        self.fcsize = 2

        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef

        self.main = nn.Sequential(
            nn.Conv3d(nch, 64, ksize, dstep, 1),
            nn.BatchNorm3d(64),

            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, ksize, dstep, 1),
            nn.BatchNorm3d(128),

            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, ksize, dstep, 1),
            nn.BatchNorm3d(256),

            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, ksize, dstep, 1),
            nn.BatchNorm3d(512),

            nn.ReLU(inplace=True),
            nn.Conv3d(512, 1024, ksize, dstep, 1),
            nn.BatchNorm3d(1024),

            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 1024, ksize, dstep, 1),
            nn.BatchNorm3d(1024),

            nn.ReLU(inplace=True)
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

            # self.refOutSigma = nn.Sequential(
            #     nn.Linear(1024*int(self.fcsize*1*1), self.nRef))


        if self.nLatentDim > 0:
            self.latentOutMu = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nLatentDim))

            self.latentOutSigma = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nLatentDim))

    def forward(self, x, reparameterize=False):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids

        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 1024*int(self.fcsize*1*1))

        xOut = list()

        
        if self.nClasses > 0:
            xClasses = nn.parallel.data_parallel(self.classOut, x, gpu_ids)
            xOut.append(xClasses)

        if self.nRef > 0:
            xRefMu = nn.parallel.data_parallel(self.refOutMu, x, gpu_ids)
            # xRefSigma = nn.parallel.data_parallel(self.refOutSigma, x, gpu_ids)
            xOut.append(xRefMu)

        if self.nLatentDim > 0:
            xLatentMu = nn.parallel.data_parallel(self.latentOutMu, x, gpu_ids)
            xLatentSigma = nn.parallel.data_parallel(self.latentOutSigma, x, gpu_ids)
            
            if self.training:
                xOut.append([xLatentMu, xLatentSigma])
            else:
                xOut.append(xLatentMu + xLatentSigma.div(2).exp())



        return xOut

class Dec(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, nch, gpu_ids, opt=None):
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
            nn.ConvTranspose3d(1024, 1024, ksize, dstep, 1, output_padding = (0,1,0)),
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
            spectral_norm(nn.Conv3d(nch, 64, ksize, dstep, 1)),
            # nn.BatchNorm3d(64),

            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(64, 128, ksize, dstep, 1)),
            nn.BatchNorm3d(128),

            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(128, 256, ksize, dstep, 1)),
            nn.BatchNorm3d(256),

            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(256, 512, ksize, dstep, 1)),
            nn.BatchNorm3d(512),

            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(512, 1024, ksize, dstep, 1)),
            nn.BatchNorm3d(1024),

            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(1024, 1024, ksize, dstep, 1)),
            nn.BatchNorm3d(1024),

            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = spectral_norm(nn.Linear(1024*int(self.fcsize), nout))

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
                self.noise = torch.zeros(x.size()).type_as(x)

            #sample random noise
            self.noise.normal_(mean=0, std=self.noise_std)
            noise = torch.autograd.Variable(self.noise)

            #add to input
            x = x + noise

        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 1024*int(self.fcsize))
        x = self.fc(x)

        return x
