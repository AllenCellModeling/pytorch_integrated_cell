from torch import nn
import pdb

ksize = 4
dstep = 2

class Enc(nn.Module):
    def __init__(self, nlatentdim, insize):
        super(Enc, self).__init__()
        
        self.fcsize = (insize/64) * 4
        
        self.conv1 = nn.Conv2d(3, 64, ksize, dstep, 1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.nl1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 128, ksize, dstep, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.nl2 = nn.PReLU()
        self.conv3 = nn.Conv2d(128, 256, ksize, dstep, 1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.nl3 = nn.PReLU()
        self.conv4 = nn.Conv2d(256, 512, ksize, dstep, 1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.nl4 = nn.PReLU()
        
        self.linear5 = nn.Linear(512*int(self.fcsize**2), nlatentdim)
        self.bn5 = nn.BatchNorm1d(nlatentdim)
        
    def forward(self, x):
        
        # pdb.set_trace()
        
        x = self.nl1(self.bn1(self.conv1(x)))
        x = self.nl2(self.bn2(self.conv2(x)))
        x = self.nl3(self.bn3(self.conv3(x)))
        x = self.nl4(self.bn4(self.conv4(x)))
        
        x = x.view(x.size()[0], 512*int(self.fcsize**2))
        x = self.bn5(self.linear5(x))
    
        return x
    
class Dec(nn.Module):
    def __init__(self, nlatentdim, insize):
        super(Dec, self).__init__()
        
        self.fcsize = int((insize/64) * 4)
        
        self.linear1 = nn.Linear(nlatentdim, 512*int(self.fcsize**2))
        self.bn1 = nn.BatchNorm1d(512*int(self.fcsize**2))
        
        self.nl1 = nn.PReLU()        
        self.conv2 = nn.ConvTranspose2d(512, 256, ksize, dstep, 1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.nl2 = nn.PReLU()        
        self.conv3 = nn.ConvTranspose2d(256, 128, ksize, dstep, 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.nl3 = nn.PReLU()        
        self.conv4 = nn.ConvTranspose2d(128, 64, ksize, dstep, 1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.nl4 = nn.PReLU()        
        self.conv5 = nn.ConvTranspose2d(64, 3, ksize, dstep, 1)
        # self.bn5 = nn.BatchNorm2d(3)
        self.nl5 = nn.Sigmoid()              
            
    def forward(self, x):
        
        x = self.bn1(self.linear1(x))
        x = x.view(x.size()[0], 512, self.fcsize, self.fcsize)

        x = self.nl1(x)
        x = self.nl2(self.bn2(self.conv2(x)))
        x = self.nl3(self.bn3(self.conv3(x)))
        x = self.nl4(self.bn4(self.conv4(x)))
        x = self.nl5(self.conv5(x))
        
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
    

