import torch
from torch import nn
from torch.nn import functional as F
from .basic_module import BasicModule

'''
Sub Structure: Bottleneck
'''
class bottleNeck(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1, downsample = None, expansion = 1 ):
        super(bottleNeck,self).__init__()
        self.stride = stride

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, inchannel*expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(inchannel*expansion),
            nn.ReLU6(inplace=True),

            nn.Conv2d(inchannel*expansion, inchannel*expansion, kernel_size=3, stride=stride,
                      padding= 1, bias=False, groups=inchannel*expansion),
            nn.BatchNorm2d(inchannel*expansion),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inchannel*expansion, outchannel, kernel_size=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )

        self.downsample = downsample

    def forward(self,x):
        out = self.left(x)
        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        out = F.relu6(out)
        return out



class MobilenetV2(BasicModule):
    '''
    This is the main Module MobileNetV2
    '''

    def __init__(self,num_classes = 2):
        super(MobilenetV2,self).__init__()
        self.inchannel = 32
        self.pre = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        ##bottleNeck Repeat Structure
        self.layer1 = self._make_layer(32,16,1,1,1)
        self.layer2 = self._make_layer(16,24,2,2,6)
        self.layer3 = self._make_layer(24,32,3,2,6)
        self.layer4 = self._make_layer(32,64,4,2,6)
        self.layer5 = self._make_layer(64,96,3,1,6)
        self.layer6 = self._make_layer(96,160,3,2,6)
        self.layer7 = self._make_layer(160,320,1,1,6)
        self.post = nn.Sequential(
            nn.Conv2d(320,1280,kernel_size=1,stride=1,bias=False),
            nn.AvgPool2d(7,stride=1),
            nn.Conv2d(1280,num_classes,kernel_size=1,stride=1,bias = False))


    def _make_layer(self,inchannel,outchannel,n,stride = 1,expansion = 1 ):
        '''
        Make Layer (which contains several bottleneck structure)
        '''
        downsample = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(outchannel),
        )

        layers = []
        layers.append(bottleNeck(inchannel,outchannel,stride,downsample=downsample,expansion=expansion))

        for i in range(1,n):
            layers.append(bottleNeck(outchannel,outchannel,expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.post(x)
        x = x.view(x.size(0), -1)

        return x

