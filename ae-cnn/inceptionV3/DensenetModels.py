import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision
from torch.autograd import Function
import torch.nn.functional as F

import re
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import time


# -----AECNN  - class for InceptionV3 type model models which need upscaling of encoder output (224x224) before passing to classifier requiring 299x299
# -----AECNN0 - class for resnets and densenets type models requiring 224x224 images 
class AECNN(nn.Module):

    def __init__(self, classCount):
        super (AECNN, self).__init__()

        self.classCount = classCount
        # self.y2 = torch.Tensor(bs, 3, h, w).cuda()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize(299)
        self.to_tensor = transforms.ToTensor()

        self.encoder = nn.Sequential(
            #1x896x896
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 4, padding = 2),
            nn.ELU(),
            #1X224X224
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
            #1x224x224
            )

        self.decoder = nn.Sequential(
            #1x224x224
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            #1x896x896
            )


        #CLASSIFIER
        self.classifier = InceptionV3(classCount = self.classCount, isTrained = True)

    def forward(self, x):
        
        y = self.encoder(x)
        y = Relu1.apply(y)
        
        z1 = self.decoder(y)
        z1 = Relu1.apply(z1)
        
        bs, c, h, w = y.shape
        y2 = torch.Tensor(bs, 3, 299, 299).cuda()
        
        for img_no in range(bs):
            y[img_no] = self.to_pil(y[img_no])
            y[img_no] = self.resize(y[img_no])
            y[img_no] = self.to_tensor(y[img_no])
            y2[img_no] = y[img_no]  #broadcasting 1 channel to 3 channels
            #y2[img_no] = self.normalize(y2[img_no]) #inception does normalising internally/automatically

        z2 = self.classifier(y2)

        return z1, z2


class AECNN0(nn.Module):

    def __init__(self, classCount):
        super (AECNN0, self).__init__()

        self.classCount = classCount
        # self.y2 = torch.Tensor(bs, 3, h, w).cuda()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.encoder = nn.Sequential(
            #1x896x896
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 4, padding = 2),
            nn.ELU(),
            #1X224X224
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
            #1x224x224
            )

        self.decoder = nn.Sequential(
            #1x224x224
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            #1x896x896
            )



        self.classifier = DenseNet121(classCount = self.classCount, isTrained = True)

    def forward(self, x):
        
        y = self.encoder(x)
        y = Relu1.apply(y)
        
        z1 = self.decoder(y)
        z1 = Relu1.apply(z1)
        
        bs, c, h, w = y.shape
        y2 = torch.Tensor(bs, 3, h, w).cuda()
        
        for img_no in range(bs):
            y2[img_no] = y[img_no]
            y2[img_no] = self.normalize(y2[img_no]) #broadcasting 1 channel to 3 channels

        z2 = self.classifier(y2)

        return z1, z2



class Relu1(Function):

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        #print("fwd:", input[0])
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0]*=0.0
        grad_input[input>1]*=0.0

        return grad_input











#RESNETS

class Resnet18(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet18, self).__init__()
        
        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)

        kernelCount = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet18(x)
        return x


class Resnet34(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet34, self).__init__()
        
        self.resnet34 = torchvision.models.resnet34(pretrained=isTrained)

        kernelCount = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet34(x)
        return x


class Resnet50(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet50, self).__init__()
        
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        kernelCount = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x


class Resnet101(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet101, self).__init__()
        
        self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)

        kernelCount = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet101(x)
        return x


class Resnet152(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Resnet152, self).__init__()
        
        self.resnet152 = torchvision.models.resnet152(pretrained=isTrained)

        kernelCount = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet152(x)
        return x






#DENSENETS

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(DenseNet121, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class DenseNet161(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(DenseNet161, self).__init__()
        
        self.densenet161 = torchvision.models.densenet161(pretrained=isTrained)

        kernelCount = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet161(x)
        return x


class DenseNet169(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)

        kernelCount = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet169(x)
        return x




class DenseNet201(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)

        kernelCount = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet201(x)
        return x






#INCEPTION

class InceptionV3(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(InceptionV3, self).__init__()
        
        self.inceptionv3 = torchvision.models.inception_v3(pretrained=isTrained)
        self.inceptionv3.transform_input = True

        kernelCount = self.inceptionv3.AuxLogits.fc.in_features
        self.inceptionv3.AuxLogits.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
        kernelCount = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.inceptionv3(x)
        return x
