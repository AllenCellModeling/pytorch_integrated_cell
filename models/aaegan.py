from torch import nn
import torch
import pdb

ksize = 4
dstep = 2

class Enc(nn.Module):
    def __init__(self, nlatentdim, insize, gpu_ids):
        super(Enc, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = (insize/64) * 2
        
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(64),
        
            nn.ELU(),
            nn.Conv2d(64, 128, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(128),
        
            nn.ELU(),
            nn.Conv2d(128, 256, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(256),
        
            nn.ELU(),
            nn.Conv2d(256, 512, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(512),
            
            nn.ELU(),
            nn.Conv2d(512, 1024, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(1024),
        
            nn.ELU()
        )
        
        self.fc = nn.Linear(1024*int(self.fcsize**2), nlatentdim)
        self.bnEnd = nn.BatchNorm1d(nlatentdim)
        
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
            
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        
        x = x.view(x.size()[0], 1024*int(self.fcsize**2))
        x = self.bnEnd(self.fc(x))
    
        return x
    
class Dec(nn.Module):
    def __init__(self, nlatentdim, insize, gpu_ids):
        super(Dec, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = int((insize/64) * 2)
        
        self.fc = nn.Linear(nlatentdim, 1024*int(self.fcsize**2))
        self.main = nn.Sequential(
            nn.BatchNorm2d(1024),
        
            nn.ELU(),
            nn.ConvTranspose2d(1024, 512, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(512),
            
            nn.ELU(),
            nn.ConvTranspose2d(512, 256, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(256),
        
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(128),
        
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(64),
        
            nn.ELU(),
            nn.ConvTranspose2d(64, 3, ksize, dstep, 1, bias=False),
            # self.bn5 = nn.BatchNorm2d(3)
            nn.Sigmoid()             
        )
            
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        x = self.fc(x)
        x = x.view(x.size()[0], 1024, self.fcsize, self.fcsize)
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
 
        return x    
    
class EncD(nn.Module):
    def __init__(self, nlatentdim, gpu_ids):
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
        
            nn.Linear(512, 1)
            nn.Sigmoid()
        )
#         self.bn3 = nn.BatchNorm1d(1)        
        # self.nl3 = nn.Sigmoid()         
          
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
            
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x.view(1)
        # x = x.mean(0).view(1)
        
        return x        

class DecD(nn.Module):
    def __init__(self, nout, insize, gpu_ids):
        super(DecD, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = int((insize/64) * 2)
        
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, ksize, dstep, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
            
        )
        
        self.fc = nn.Linear(1024*int(self.fcsize**2), nout)
        self.nlEnd = nn.Sigmoid()
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
        x = x.view(x.size()[0], 1024*int(self.fcsize**2))
        x = self.fc(x)
        x = self.nlEnd(x)

        return x.view(1)
    

