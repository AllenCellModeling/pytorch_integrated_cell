from torch import nn
import torch
import pdb
from model_utils import init_opts

ksize = 4
dstep = 2

class Enc(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, insize, nch, gpu_ids, opt=None):
        super(Enc, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = insize/64
        
        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef
        
        self.main = nn.Sequential(
            nn.Conv2d(nch, 64, ksize, dstep, 1),
            nn.BatchNorm2d(64),
        
            nn.PReLU(),
            nn.Conv2d(64, 128, ksize, dstep, 1),
            nn.BatchNorm2d(128),
        
            nn.PReLU(),
            nn.Conv2d(128, 256, ksize, dstep, 1),
            nn.BatchNorm2d(256),
        
            nn.PReLU(),
            nn.Conv2d(256, 512, ksize, dstep, 1),
            nn.BatchNorm2d(512),
            
            nn.PReLU(),
            nn.Conv2d(512, 1024, ksize, dstep, 1),
            nn.BatchNorm2d(1024),
            
            nn.PReLU(),
            nn.Conv2d(1024, 1024, ksize, dstep, 1),
            nn.BatchNorm2d(1024),
        
            nn.PReLU()
        )
        
        if self.nClasses > 0:
            self.classOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize**2), self.nClasses),
                nn.BatchNorm1d(self.nClasses),
                nn.LogSoftmax()
            )
        
        if self.nRef > 0:
            self.refOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize**2), self.nRef),
                nn.BatchNorm1d(self.nRef)
            )
        
        if self.nLatentDim > 0:
            self.latentOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize**2), self.nLatentDim),
                nn.BatchNorm1d(self.nLatentDim)
            )
            
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
            
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 1024*int(self.fcsize**2))
        
        xOut = list()
                
        if self.nClasses > 0:
            xClasses = nn.parallel.data_parallel(self.classOut, x, gpu_ids)
            xOut.append(xClasses)
            
        if self.nRef > 0: 
            xRef = nn.parallel.data_parallel(self.refOut, x, gpu_ids)
            xOut.append(xRef)
        
        if self.nLatentDim > 0:
            xLatent = nn.parallel.data_parallel(self.latentOut, x, gpu_ids)
            xOut.append(xLatent)
        
        return xOut
    
class Dec(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, insize, nch, gpu_ids, opt=None):
        super(Dec, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = int(insize/64)
        
        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef
        
        self.fc = nn.Linear(self.nLatentDim + self.nClasses + self.nRef, 1024*int(self.fcsize**2))
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(1024),
            
            nn.PReLU(),
            nn.ConvTranspose2d(1024, 1024, ksize, dstep, 1),
            nn.BatchNorm2d(1024),
            
            nn.PReLU(),
            nn.ConvTranspose2d(1024, 512, ksize, dstep, 1),
            nn.BatchNorm2d(512),
            
            nn.PReLU(),
            nn.ConvTranspose2d(512, 256, ksize, dstep, 1),
            nn.BatchNorm2d(256),
        
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, ksize, dstep, 1),
            nn.BatchNorm2d(128),
        
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, ksize, dstep, 1),
            nn.BatchNorm2d(64),
        
            nn.PReLU(),
            nn.ConvTranspose2d(64, nch, ksize, dstep, 1),
            # nn.BatchNorm2d(nch),
            nn.Sigmoid()             
        )
            
    def forward(self, xIn):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        x = torch.cat(xIn, 1)
        
        x = self.fc(x)
        x = x.view(x.size()[0], 1024, self.fcsize, self.fcsize)
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
 
        return x    
    
class EncD(nn.Module):
    def __init__(self, nlatentdim, gpu_ids, opt=None):
        super(EncD, self).__init__()
        
        nfc = 1024
        
        self.gpu_ids = gpu_ids
        
        self.main = nn.Sequential(
            nn.Linear(nlatentdim, nfc),
            nn.BatchNorm1d(nfc),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Linear(nfc, nfc),
            nn.BatchNorm1d(nfc),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Linear(nfc, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
            
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)

        return x        

class DecD(nn.Module):
    def __init__(self, nout, insize, nch, gpu_ids, opt=None):
        super(DecD, self).__init__()
        
        def opt_default(): 1
        opt_default.noise_std = 0
        opt_default.nclasses = 1
        
        self.opt = init_opts(opt, opt_default)
        
        self.gpu_ids = gpu_ids
        self.fcsize = insize/64
        
        self.noise = torch.zeros(0)
        
        self.main = nn.Sequential(
            nn.Conv2d(nch, 64, ksize, dstep, 1),
            # nn.BatchNorm2d(64),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, ksize, dstep, 1),
            nn.BatchNorm2d(128),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, ksize, dstep, 1),
            nn.BatchNorm2d(256),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, ksize, dstep, 1),
            nn.BatchNorm2d(512),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, ksize, dstep, 1),
            nn.BatchNorm2d(512),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, ksize, dstep, 1),
            nn.BatchNorm2d(512),
            
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Linear(512*int(self.fcsize**2), nout)

        if nout == 1:
            self.nlEnd = nn.Sigmoid()
        else:
            self.nlEnd = nn.LogSoftmax()
            
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        if self.opt.noise > 0:
            #allocate an appropriately sized variable if it does not exist
            if self.noise.size() != x.size():
                self.noise = torch.zeros(x.size()).cuda(self.gpu_ids[0])

            #sample random noise
            self.noise.normal_(mean=0, std=1)
            noise = torch.autograd.Variable(self.noise)

            #add to input
            x = x + noise
        
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 512*int(self.fcsize**2))
        x = self.fc(x)
        x = self.nlEnd(x)

        return x
    

