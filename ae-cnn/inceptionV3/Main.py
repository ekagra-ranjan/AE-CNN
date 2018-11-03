import os
import numpy as np
import time
import sys
import torch

import torch.backends.cudnn as cudnn
#cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
    parentPath = '/home/histosr/ChestX'
    #parentPath = '/home/siplab-15/ChestX'
    pathDirData = parentPath+'/database_new/train_set'
    pathDirDataTest = parentPath+'/database_new/test_set'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = parentPath+'/database_new/train_only_14.txt'
    pathFileVal = parentPath+'/database_new/val_only_14.txt'
    pathFileTest = parentPath+'/database_new/test_final_14.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = 'Autoencoder-pE-pD-pinc82'  #82 denotes the weighting bw main and aux outputs
    nnIsTrained = True
    nnClassCount = 14
    
    
    #---- Training settings: batch size, maximum number of epochs
    trainSize = 76524
    trBatchSize = 8
    trMaxEpoch = 15
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    transResize = 256

    transCrop = 896
    threshold = torch.Tensor([0.5])
    threshold = threshold.cuda()
    

    # ----- pathModel - path to the AE-CNN model which is to be tested 
    # -----checkpoint1 - is initialised with a  dummy. If not none then it loads weights of encoder and decoder from "encoder.pth" and "decoder.pth" seperately into AE-CNN encoder and decoder
    # -----checkpoint2 - if not none, loads the classifier of AE-CNN from pretrained classifier on ChestX-Ray14 dataset
    # -----checkpoint3 - if not none, loads the full AE-CNN weights for resuming the training from a saved instance
    pathModel = None
    # pathModel = 'm-Autoencoder-pE-pD-pd121-epoch-15-auc-0.843020353288727-16072018-233246.pth.tar'

    checkpoint1 = 'm-Autoencoder-epoch-17-loss-0.00026260194169445625-15072018-133705.pth.tar'
    checkpoint2 = 'm-INCEPTION-V3-epoch-6-auc-0.8373397258160715-19072018-092543.pth.tar'
    #checkpoint3 = 'm-Autoencoder-pE-pD-pd121-epoch-14-auc-0.8429981074471875-16072018-233246.pth.tar'
    checkpoint3 = None

    print ('Training NN architecture = ', nnArchitecture)
    pathModel = ChexnetTrainer.train(parentPath, pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, timestampLaunch, threshold, device, checkpoint1, checkpoint2, checkpoint3)
    print("Path Model: "+str(pathModel))

    # -----ChexnetTrainer.store() - function to store the output of encoder of AE-CNN. This can be used for visualisation of latent code produced by encoder
    # ChexnetTrainer.store(parentPath, pathModel, nnArchitecture, nnClassCount, trBatchSize, device)

    # -----uncomment the below block and comment out the training block for testin the AE-CNN from pathModel.
    # print ('Testing the trained model')
    # ChexnetTrainer.test(parentPath, None, pathModel, pathDirDataTest, pathFileTest, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, timestampLaunch, threshold, device)

#-------------------------------------------------------------------------------- 


if __name__ == '__main__':
    main()

