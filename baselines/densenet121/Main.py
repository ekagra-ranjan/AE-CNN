import os
import numpy as np
import time
import sys
import torch

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

from ChexnetTrainer import ChexnetTrainer



#-------------------------------------------------------------------------------- 

def main ():
    
    runTrain()
    #runTest()
  
#--------------------------------------------------------------------------------   

def runTrain():

    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)
    
    RESNET18 = 'RES-NET-18'
    RESNET34 = 'RES-NET-34'
    RESNET50 = 'RES-NET-50'
    RESNET101 = 'RES-NET-101'
    RESNET152 = 'RES-NET-152'    
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET161 = 'DENSE-NET-161'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'

    INCEPTIONV3 = 'INCEPTION-V3'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    parentPath = '/home/deepsip/ChestX'
    #parentPath = '/home/siplab-15/ChestX'
    pathDirData = './database/train_set'
    pathDirDataTest ='./database/test_set'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_final.txt'
    pathFileTest = './dataset/test_final.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 15
    
    
    #---- Training settings: batch size, maximum number of epochs
    trainSize = 76524
    trBatchSize = 16   
    trMaxEpoch = 50
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    transResize = 256
    if nnArchitecture == 'INCEPTION-V3':
        transCrop = 299
    else:
        transCrop = 224
    threshold = torch.Tensor([0.5])
    threshold = threshold.cuda()
       
        
    #pathModel = 'm-INCEPTION-V3-epoch-11-auc-0.8287996946829962-10062018-151252.pth.tar'
    #pathModel = 'm-DENSE-NET-121-epoch-19-auc-0.8335507109376249-07062018-204359.pth.tar'
    #pathModel = 'm-DENSE-NET-161-epoch-19-auc-0.8350557904266428-08062018-174400.pth.tar'
    #pathModel = 'm-RES-NET-152-epoch-31-auc-0.8286113250937469-08062018-164857.pth.tar'
    #pathModel = 'm-INCEPTION-V3-epoch-9-auc-0.8311135682198351-13062018-141146.pth.tar'
    pathModel = 'm-RES-NET-18-epoch-11-auc-0.8293945115946493-17062018-162956.pth.tar'

    #checkpoint = 'm-INCEPTION-V3-epoch-3-04062018-193255.pth.tar'
    #checkpoint = 'm-INCEPTION-V3-epoch-12-auc-0.8212759086083358-11062018-125118.pth.tar'
    #checkpoint = 'm-RES-NET-18-epoch-19-auc-0.8224725440125409-10062018-200406.pth.tar'
    checkpoint = None


    #pathModel = 'm-DENSE-NET-121-epoch-3-29052018-170005.pth.tar'
    #pathModel="m-DENSE-NET-121-epoch-19-auc-0.8335507109376249-07062018-204359.pth.tar"
    #checkpoint="m-DENSE-NET-121-epoch-19-auc-0.8335507109376249-07062018-204359.pth.tar"

    print("No of CLasses:", nnClassCount)

    # print ('Training NN architecture = ', nnArchitecture)
    # pathModel = ChexnetTrainer.train(parentPath, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, timestampLaunch, threshold, device, checkpoint)
    # print("Path Model: "+str(pathModel))
    
    print ('Testing the trained model')
    ChexnetTrainer.test(parentPath, None, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, timestampLaunch, threshold, device)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = './database'
    pathFileTest = './dataset/test_1.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 15
    trBatchSize = 32
    imgtransResize = 256
    imgtransCrop = 224
    

    pathModel = './models/m-25012018-123527.pth.tar'
    
    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()

