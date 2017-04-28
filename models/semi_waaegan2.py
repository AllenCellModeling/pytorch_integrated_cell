from torch import nn
import pdb

ksize = 4
dstep = 2

class Enc(nn.Module):
    def __init__(self, nlatentdim, insize):
        super(Enc, self).__init__()
        
        
        
        self.fcsize = (insize/64) * 4
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, ksize, dstep, 1),
            # nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, ksize, dstep, 1),
            # nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, ksize, dstep, 1),
            # nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 512, ksize, dstep, 1),
            # nn.BatchNorm2d(512)
            nn.ELU(inplace=True)
        )
        
        self.fc = nn.Linear(512*int(self.fcsize**2), nlatentdim)
        
        
    def forward(self, x):
        
        x = self.model(x)
        x = x.view(x.size()[0], 512*int(self.fcsize**2))
        x = self.fc(x)
    
        return x
    
class Dec(nn.Module):
    def __init__(self, nlatentdim, insize):
        super(Dec, self).__init__()
        
        self.fcsize = int((insize/64) * 4)
        
        
        self.fc = nn.Linear(nlatentdim, 512*int(self.fcsize**2))
        self.bn1 = nn.BatchNorm1d(512*int(self.fcsize**2))
        
        
        self.model = nn.Sequential(
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(512, 256, ksize, dstep, 1),
            # nn.BatchNorm2d(256),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(256, 128, ksize, dstep, 1),
            # nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(128, 64, ksize, dstep, 1),
            # nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(64, 3, ksize, dstep, 1),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        
        x = self.fc(x)
        # x = self.bn1(x)
        x = x.view(x.size()[0], 512, self.fcsize, self.fcsize)
        x = x.model(x)
        
        return x    
    
class EncD(nn.Module):
    def __init__(self, nlatentdim):
        super(EncD, self).__init__()
        
        nfc = 1024
        
        self.model = nn.Sequential(
            nn.Linear(nlatentdim, nfc),
            # nn.BatchNorm1d(nfc),        
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nfc, nfc),
            # nn.BatchNorm1d(nfc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nfc, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )
          
    def forward(self, x):
        x = self.model(x)
        return x        

class DecD(nn.Module):
    def __init__(self, nout, insize):
        super(DecD, self).__init__()
        
        self.fcsize = int((insize/64) * 4)
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, ksize, dstep, 1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, ksize, dstep, 1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, ksize, dstep, 1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, ksize, dstep, 1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc = nn.Linear(512*int(self.fcsize**2), nout)
    def forward(self, x):

        x = self.model(x)
        x = x.view(x.size()[0], 512*int(self.fcsize**2))
        x = self.fc(x)
        # x = x.mean(0)
        # x = x.view(1)
        return x
    

