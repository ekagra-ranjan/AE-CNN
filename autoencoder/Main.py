import os
import numpy as np
import time
import sys
import torch

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
    nnArchitecture = 'Autoencoder'
    nnIsTrained = True
    nnClassCount = 14
    
    
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
       
    

    
    checkpoint = None

    print ('Training NN architecture = ', nnArchitecture)
    pathModel = ChexnetTrainer.train(parentPath, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, timestampLaunch, threshold, device, checkpoint)
    print("Path Model: "+str(pathModel))
    # print ('Testing the trained model')
    # ChexnetTrainer.test(parentPath, None, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, timestampLaunch, threshold, device)

#-------------------------------------------------------------------------------- 


if __name__ == '__main__':
    main()

