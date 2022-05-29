from torch import nn
import torch as t
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
                    nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
                    nn.BatchNorm2d(outchannel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
                    nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self,x):
        out = self.left(x)
        # print("------------test point-------------")
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
class ResidualBlock2(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock2,self).__init__()
        self.left_c1 = nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False)
        self.left_b1 = nn.BatchNorm2d(outchannel)
        self.left_r1 = nn.ReLU(inplace=True)
        self.left_c2 = nn.Conv2d(outchannel,outchannel,3,1,1,bias=False)
        self.left_b2 = nn.BatchNorm2d(outchannel)
        self.left_r2 = nn.ReLU(inplace=True)
        self.left_c3 = nn.Conv2d(outchannel,outchannel,3,1,1,bias=False)
        self.left_b3 = nn.BatchNorm2d(outchannel)
        self.left_r3 = nn.ReLU(inplace=True)
        self.left_c4 = nn.Conv2d(outchannel,outchannel,3,1,1,bias=False)
        self.left_b4 = nn.BatchNorm2d(outchannel)
        self.right = shortcut

    def forward(self,xt):
        x = self.left_c1(xt)
        x = self.left_b1(x)
        cx = self.left_r1(x)
        x = self.left_c2(cx)
        out1 = self.left_b2(x)
        
        x = self.left_r2(out1+xt)
        x = self.left_c3(x)
        x = self.left_b3(x)
        x = self.left_r3(x+cx)
        x = self.left_c4(x)
        out2 = self.left_b4(x)

        return F.relu(out2+out1)

class ResNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ResNet,self).__init__()

        self.pre = nn.Sequential(
                   nn.Conv2d(3,64,7,2,3,bias=False),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True),
                   nn.MaxPool2d(3,2,1))
        
        #=============layer 1================================
        shortcut = nn.Sequential(
                   nn.Conv2d(64,128,1,1,bias=False),
                   nn.BatchNorm2d(128))   
        self.layer1_p1= ResidualBlock(64,128,1,shortcut)
        self.layer1_p2= ResidualBlock(128,128)
        self.layer1_p3= ResidualBlock2(128,128)
        # self.layer1_p4= ResidualBlock(128,128)
        #=============layer 2================================
        self.layer2 = self._make_layer(128,256,4,stride=2)
        self.layer3 = self._make_layer(256,512,6,stride=2)
        self.layer4 = self._make_layer(512,512,3,stride=2)
        self.fc = nn.Linear(512,num_classes)
        
    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        shortcut = nn.Sequential(
                   nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
                   nn.BatchNorm2d(outchannel))      
        layers = []
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self,x):
        print("------------x.shape0=",x.shape)
        x = self.pre(x)
        #=============layer 1============================
        x = self.layer1_p1(x)
        x = self.layer1_p2(x)
        x = self.layer1_p3(x)
        # x = self.layer1_p4(x)

        # print("------------x.shape2=",x.shape)
        x = self.layer2(x)
        # print("------------x.shape3=",x.shape)
        x = self.layer3(x)
        # print("------------x.shape4=",x.shape)
        x = self.layer4(x)
        # print("------*******************------x.shape5=",x.shape)
        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0),-1)
        return self.fc(x)

# model = ResNet()

# input = r.autograd.Variable(t,randn(1,3,224,224))

# o = model(input)