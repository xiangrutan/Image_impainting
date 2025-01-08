import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
import torchvision.models as models

# we do conv, batchnorm, relu twice
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)

class MyUNet_DIP(nn.Module):
    def __init__(self, hole_size, criterion=None, bilinear = False):
        super(MyUNet_DIP, self).__init__()
        self.criterion = criterion
        self.hole_size = hole_size
        
        # init
        self.init = (DoubleConv(3, 64))
        
        # downsample 1
        self.ds1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128))
        
        # downsample 2
        self.ds2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256))
                
        # downsample 3
        self.ds3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512))

        # downsample 4
        self.ds4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024))
        
        if bilinear:
            # upsample 1
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv1 = DoubleConv(1536, 512)

            # upsample 2
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv2 = DoubleConv(768, 256)

            # upsample 3
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv3 = DoubleConv(384, 128)

            # upsample 4
            self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv4 = DoubleConv(192, 64) 
        else:
            # upsample 1
            self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.upconv1 = DoubleConv(1024, 512)

            # upsample 2
            self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.upconv2 = DoubleConv(512, 256)

            # upsample 3
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.upconv3 = DoubleConv(256, 128)

            # upsample 4
            self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.upconv4 = DoubleConv(128, 64)

        # last layer
        self.out = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, inp, gts=None):

        inp_shape = inp.size()
                
        init = self.init(inp)
        ds1 = self.ds1(init)
        ds2 = self.ds2(ds1)
        ds3 = self.ds3(ds2)
        ds4 = self.ds4(ds3)

        lfinal = self.up1(ds4)
        lfinal = tF.resize(lfinal, (ds3.size()[2], ds3.size()[3]))
        lfinal = torch.cat((ds3, lfinal), dim = 1)
        lfinal = self.upconv1(lfinal)

        lfinal = self.up2(lfinal)
        lfinal = tF.resize(lfinal, (ds2.size()[2], ds2.size()[3]))
        lfinal = torch.cat((ds2, lfinal), dim = 1)
        lfinal = self.upconv2(lfinal)

        lfinal = self.up3(lfinal)
        lfinal = tF.resize(lfinal, (ds1.size()[2], ds1.size()[3]))
        lfinal = torch.cat((ds1, lfinal), dim = 1)
        lfinal = self.upconv3(lfinal)

        lfinal = self.up4(lfinal)
        lfinal = tF.resize(lfinal, (init.size()[2], init.size()[3]))
        lfinal = torch.cat((init, lfinal), dim = 1)
        lfinal = self.upconv4(lfinal)

        lfinal = tF.resize(lfinal, (inp_shape[2], inp_shape[3]))
        lfinal = self.out(lfinal)
        
        if self.training:
            # Return the loss if in training mode
            wlh = inp_shape[3]//2 - self.hole_size//2
            wrh = inp_shape[3]//2 + self.hole_size//2
            wlw = inp_shape[2]//2 - self.hole_size//2
            wrw = inp_shape[2]//2 + self.hole_size//2
            lfinal[:,wlw:wrw,wlh:wrh] = gts[:,wlw:wrw,wlh:wrh]
            return self.criterion(lfinal, gts), lfinal             
        else:
            # Return the actual prediction otherwise
            return lfinal

class MyNet_DIP(nn.Module):

    def __init__(self, hole_size, criterion=None):
        super(MyNet_DIP, self).__init__()
        self.criterion = criterion
        self.hole_size = hole_size
        
        # Resnet18
        self.resnet18 = models.resnet18(pretrained = True)
        
        # upsample 1
        # self.upsample1 = nn.ConvTranspose2d(512, 64, 13, stride=8, padding=0)
                        
        # some random cn
        self.conv1 = nn.Conv2d(576, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        # upsample 2
        self.upsample2 = nn.ConvTranspose2d(64, 3, 13, stride=8, padding=0)

    
    def forward(self, inp, gts=None):
                
        inp_shape = inp.size()
        
        # ResNet18
        inp = self.resnet18.conv1(inp)      
        inp = self.resnet18.bn1(inp)
        inp = self.resnet18.relu(inp)
        inp = self.resnet18.maxpool(inp)
        inp = self.resnet18.layer1(inp)
        save_concat = inp
        inp = self.resnet18.layer2(inp)
        inp = self.resnet18.layer3(inp)
        inp = self.resnet18.layer4(inp)
                
        # upsample 1
        inp = nn.Upsample(size=(save_concat.size()[2], save_concat.size()[3]), mode='bilinear', align_corners=True)(inp)
        
        # change to the same size as layer 1 and concat
        inp = torch.cat((inp, save_concat), dim = 1)
        
        # some random c
        inp = self.conv1(inp)
        inp = F.relu(inp, inplace=True)
        inp = self.conv2(inp)
        inp = F.relu(inp, inplace=True)
                
        # upsample 2
        inp = self.upsample2(inp)
        
        # change to the same size as input
        inp = tF.resize(inp, (inp_shape[2], inp_shape[3]))
        lfinal = inp
        if self.training:
                # Return the loss if in training mode
                wlh = inp_shape[3]//2 - self.hole_size//2
                wrh = inp_shape[3]//2 + self.hole_size//2
                wlw = inp_shape[2]//2 - self.hole_size//2
                wrw = inp_shape[2]//2 + self.hole_size//2
                lfinal[:,wlw:wrw,wlh:wrh] = gts[:,wlw:wrw,wlh:wrh]
                return self.criterion(lfinal, gts), lfinal                    
        else:
            # Return the actual prediction otherwise
            return lfinal

class MyUNet(nn.Module):
    def __init__(self, hole_size, criterion=None, bilinear = False):
        super(MyUNet, self).__init__()
        self.criterion = criterion
        self.hole_size = hole_size
        
        # init
        self.init = (DoubleConv(3, 64))
        
        # downsample 1
        self.ds1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128))
        
        # downsample 2
        self.ds2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256))
                
        # downsample 3
        self.ds3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512))

        # downsample 4
        self.ds4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024))
        
        if bilinear:
            # upsample 1
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv1 = DoubleConv(1536, 512)

            # upsample 2
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv2 = DoubleConv(768, 256)

            # upsample 3
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv3 = DoubleConv(384, 128)

            # upsample 4
            self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upconv4 = DoubleConv(192, 64) 
        else:
            # upsample 1
            self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.upconv1 = DoubleConv(1024, 512)

            # upsample 2
            self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.upconv2 = DoubleConv(512, 256)

            # upsample 3
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.upconv3 = DoubleConv(256, 128)

            # upsample 4
            self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.upconv4 = DoubleConv(128, 64)

        # last layer
        self.out = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, inp, gts=None):

        inp_shape = inp.size()
                
        init = self.init(inp)
        ds1 = self.ds1(init)
        ds2 = self.ds2(ds1)
        ds3 = self.ds3(ds2)
        ds4 = self.ds4(ds3)

        lfinal = self.up1(ds4)
        lfinal = tF.resize(lfinal, (ds3.size()[2], ds3.size()[3]))
        lfinal = torch.cat((ds3, lfinal), dim = 1)
        lfinal = self.upconv1(lfinal)

        lfinal = self.up2(lfinal)
        lfinal = tF.resize(lfinal, (ds2.size()[2], ds2.size()[3]))
        lfinal = torch.cat((ds2, lfinal), dim = 1)
        lfinal = self.upconv2(lfinal)

        lfinal = self.up3(lfinal)
        lfinal = tF.resize(lfinal, (ds1.size()[2], ds1.size()[3]))
        lfinal = torch.cat((ds1, lfinal), dim = 1)
        lfinal = self.upconv3(lfinal)

        lfinal = self.up4(lfinal)
        lfinal = tF.resize(lfinal, (init.size()[2], init.size()[3]))
        lfinal = torch.cat((init, lfinal), dim = 1)
        lfinal = self.upconv4(lfinal)

        lfinal = tF.resize(lfinal, (inp_shape[2], inp_shape[3]))
        lfinal = self.out(lfinal)
        
        if self.training:
            # Return the loss if in training mode
            return self.criterion(lfinal, gts), lfinal             
        else:
            # Return the actual prediction otherwise
            return lfinal

class MyDis(nn.Module):

    def __init__(self, criterion=None, BATCH_SIZE = 4):
        super(MyDis, self).__init__()
        self.criterion = criterion
        self.BATCH_SIZE = BATCH_SIZE
        
        # Resnet18
        self.resnet18 = models.resnet18(pretrained = False)

        self.fc = nn.Linear(512, 1, bias = True)

    
    def forward(self, inp, gts=None):   
        # ResNet18
        inp = self.resnet18.conv1(inp)      
        inp = self.resnet18.bn1(inp)
        inp = self.resnet18.relu(inp)
        inp = self.resnet18.maxpool(inp)
        inp = self.resnet18.layer1(inp)
        inp = self.resnet18.layer2(inp)
        inp = self.resnet18.layer3(inp)
        inp = self.resnet18.layer4(inp)
        inp = self.resnet18.avgpool(inp)
        inp = torch.reshape(inp, (self.BATCH_SIZE, 512))
        lfinal = self.fc(inp)
        lfinal = torch.sigmoid(lfinal)
        if self.training:
                # Return the loss if in training mode
                return self.criterion(lfinal, gts), lfinal                    
        else:
            # Return the actual prediction otherwise
            return lfinal