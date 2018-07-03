from torch import nn
import torch
import pdb
from model_utils import init_opts

ksize = 4
dstep = 2


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, ksize, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv3d(in_channels, out_channels, ksize, stride, 1)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        # out = self.relu(out)
        return out
    
class ResidualBlockTrans(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockTrans, self).__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, ksize, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.ConvTranspose3d(in_channels, out_channels, ksize, stride, 1)
        
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        # out = self.relu(out)
        return out

    


class Enc(nn.Module):
    def __init__(self, nLatentDim, nClasses, nRef, insize, nch, gpu_ids, opt=None):
        super(Enc, self).__init__()
        
        self.gpu_ids = gpu_ids
        self.fcsize = 2
        
        self.in_channels = nch
        
        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef
        
        self.main = nn.Sequential(
            self.make_layer(64, 1, 2),
            nn.BatchNorm3d(64),
        
            nn.ReLU(),
            self.make_layer(128, 1, 2),
            nn.BatchNorm3d(128),
        
            nn.ReLU(),
            self.make_layer(256, 1, 2), 
            nn.BatchNorm3d(256),
            
            nn.ReLU(),
            self.make_layer(512, 1, 2), 
            nn.BatchNorm3d(512),
            
            nn.ReLU(),
            self.make_layer(1024, 1, 2), 
            nn.BatchNorm3d(1024),
            
            nn.ReLU(),
            self.make_layer(1024, 1, 2), 
            nn.BatchNorm3d(1024),
        
            nn.PReLU()
        )
        
        if self.nClasses > 0:
            self.classOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nClasses),
                nn.BatchNorm1d(self.nClasses),
                nn.LogSoftmax()
            )
        
        if self.nRef > 0:
            self.refOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nRef),
                nn.BatchNorm1d(self.nRef)
            )
        
        if self.nLatentDim > 0:
            self.latentOut = nn.Sequential(
                nn.Linear(1024*int(self.fcsize*1*1), self.nLatentDim),
                nn.BatchNorm1d(self.nLatentDim)
            )
            
    def make_layer(self, out_channels, blocks, stride=1):
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)       
    
    def forward(self, x):
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
        self.fcsize = 2
        
        self.in_channels = 1024
        
        self.nLatentDim = nLatentDim
        self.nClasses = nClasses
        self.nRef = nRef
        
        self.fc = nn.Linear(self.nLatentDim + self.nClasses + self.nRef, 1024*int(self.fcsize*1*1))
        
        self.main = nn.Sequential(
            nn.BatchNorm3d(1024),
            
            nn.PReLU(),
            nn.ConvTranspose3d(1024, 1024, ksize, dstep, 1, output_padding = (0,1,0)),
            nn.BatchNorm3d(1024),
            
            nn.PReLU(),
            self.make_layer(512, 1, 2),
            nn.BatchNorm3d(512),
            
            nn.PReLU(),
            self.make_layer(256, 1, 2),
            nn.BatchNorm3d(256),
            
            nn.PReLU(),
            self.make_layer(128, 1, 2),
            nn.BatchNorm3d(128),
            
            nn.PReLU(),
            self.make_layer(64, 1, 2),
            nn.BatchNorm3d(64),
            
            nn.PReLU(),
            self.make_layer(nch, 1, 2),
            # nn.BatchNorm3d(nch),
            nn.Sigmoid()             
        )
        
    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlockTrans(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResidualBlockTrans(out_channels, out_channels))
        return nn.Sequential(*layers)         
            
    def forward(self, xIn):
        # gpu_ids = None
        # if isinstance(x.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 1:
        gpu_ids = self.gpu_ids
        
        x = torch.cat(xIn, 1)
        
        x = self.fc(x)
        x = x.view(x.size()[0], 1024, self.fcsize, 1, 1)
        
        x = nn.parallel.data_parallel(self.main, x, gpu_ids)
 
        return x    
    
class EncD(nn.Module):
    def __init__(self, nlatentdim, gpu_ids, opt=None):
        super(EncD, self).__init__()
        
        nfc = 1024
        
        self.gpu_ids = gpu_ids
        
        self.main = nn.Sequential(
            nn.Linear(nlatentdim, nfc),
            # nn.BatchNorm1d(nfc),
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
            nn.Conv3d(512, 512, ksize, dstep, 1),
            nn.BatchNorm3d(512),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 512, ksize, dstep, 1),
            nn.BatchNorm3d(512),
            
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Linear(512*int(self.fcsize), nout)

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
        x = x.view(x.size()[0], 512*int(self.fcsize))
        x = self.fc(x)
        x = self.nlEnd(x)

        return x
    



