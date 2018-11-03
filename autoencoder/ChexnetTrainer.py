import os
import numpy as np
import time
from time import time as now
import sys
import random
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score
from DensenetModels import Autoencoder
#from DatasetGenerator import DatasetGenerator
#from lossFunc import lossFunc


#-------------------------------------------------------------------------------- 

class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training

    #-----------------SHUFFLING THE TENSOR-----------------------------------
    def shufflefinal(it):
        N = it.shape[0]
        for loop in range(N):

            irand1 = np.random.randint(0, N)
            irand2 = np.random.randint(0, N)

            ivar = it[irand1].clone()
            it[irand1] = it[irand2]
            it[irand2] = ivar
            

        return it


    #----TRANFORMATION FUNCTION

    #takes 256x256 and returns 224x224 (cropping)
    def trans_train(x, nnArchitecture):
        
        # if nnArchitecture != 'INCEPTION-V3':
        #     normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #normalize = transforms.Normalize([0.5057, 0.5057, 0.5057], [0.251, 0.251, 0.251])


        transformList = []
        transformList.append(transforms.ToPILImage())
        # transformList.append(transforms.RandomCrop(transCrop) )
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.RandomVerticalFlip())
        transformList.append(transforms.ToTensor())
        # if nnArchitecture != 'INCEPTION-V3':
        #     transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)
        
        y = torch.zeros( size=(x.shape), dtype = torch.float32)
        
        for i in range(x.shape[0]):
            y[i] = transformSequence(x[i])
        
        return y

    #takes 224x224/299x299 and returns 224x224/299x299 (simple testing)
    def trans_val(x, nnArchitecture):

        # if nnArchitecture != 'INCEPTION-V3':
        #     normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #normalize = transforms.Normalize([0.5057, 0.5057, 0.5057], [0.251, 0.251, 0.251])


        transformList = []
        transformList.append(transforms.ToPILImage())
        transformList.append(transforms.ToTensor())
        # if nnArchitecture != 'INCEPTION-V3':
        #     transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)
        
        y = torch.zeros_like(x, dtype = torch.float32)
        
        for i in range(x.shape[0]):
            y[i] = transformSequence(x[i])
        
        return y


    def trans_test(x, nnArchitecture, transCrop):

        # if nnArchitecture != 'INCEPTION-V3':
        #     normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #normalize = transforms.Normalize([0.5057, 0.5057, 0.5057], [0.251, 0.251, 0.251])


        transformList = []
        transformList.append(transforms.ToPILImage())   
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # if nnArchitecture != 'INCEPTION-V3':
        #     transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    
        transformSequence=transforms.Compose(transformList)
        
        y = torch.zeros( size=(x.shape[0], 10, x.shape[1], transCrop, transCrop), dtype = torch.float32)

        for i in range(x.shape[0]):
            y[i] = transformSequence(x[i])
        
        return y




    
    def train (parentPath, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, threshold , device, checkpoint):

        print("Inside train function:")
    
        #-------------------- SETTINGS: NETWORK ARCHITECTURE

        model = Autoencoder()
        #--------GPU
        #model = torch.nn.DataParallel(model)
        model.to(device)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        

        #---- Load checkpoint 
        epochStart = 0
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            epochStart = modelCheckpoint['epoch']
            print("LOADED FROM CHECKPOINT:", checkpoint)
        
    
        #--------------------RAM DATA
        print("============================ Loading data into RAM ======================================== ")

        
        trainImage = torch.load(parentPath+"/database/autoencoder_32_1948_128.pth")
     
        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000    
        print("Total No of batches : " + str(trainImage.shape[0]/trBatchSize))
        auc_per_5_epochs = []
        auc_per_epoch = []
        loss_train_per_epoch = []
        loss_val_per_epoch = []

        print('\n')
        
        for epochID in range (epochStart, trMaxEpoch):

            print("\nEpoch: ", epochID+1)
            print("LR: ", optimizer.param_groups[0]['lr'])
            
            shuffled_img = ChexnetTrainer.shufflefinal(trainImage)
            # print("===============================Shuffling done==================================")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            
        #----------------------------------------------------------------training             
            tic = now()
            lossTrain = ChexnetTrainer.trainEpoch(model, nnArchitecture, shuffled_img ,optimizer, trBatchSize, nnClassCount, threshold, device) 
            
            loss_train_per_epoch.append(lossTrain)
            
            torch.save(loss_train_per_epoch, parentPath+'/main_backup/loss/'+nnArchitecture+'-'+str(trMaxEpoch)+'-'+'loss_train_mean-'+str(launchTimestamp) )
            # print("Saved Loss for this epoch")
            
            toc = now()

            print("Time taken in this epoch {}".format(toc-tic))
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
        
            
            if ((epochID + 1)%5 != 7):

                lossMIN = lossTrain    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + nnArchitecture + '-' + 'epoch-' + str(epochID+1) + '-loss-' + str(lossTrain) + '-' + str(launchTimestamp) + '.pth.tar')
                latestModelName = 'm-' + nnArchitecture + '-' + 'epoch-' + str(epochID+1) + '-loss-' + str(lossTrain) + '-' + str(launchTimestamp) + '.pth.tar'
                print(latestModelName, " Saved")
                
                
        
        return latestModelName
                     
    #-------------------------------------------------------------------------------- 
       
        

    #--------------------------------------------------CUSTOM TRAINING WITH CUSTOM LOSS
    def trainEpoch(model, nnArchitecture, trainImage, optimizer, trBatchSize, classCount, threshold, device):

        #model.train()

       
        trainSize = trainImage.shape[0]

        loss = torch.nn.MSELoss(size_average = True)
        # print('================= Model Training starts for this epoch ===============================')

        batchID = 1

        lossTrain = 0
        lossTrainNorm = 0        
        losstensorMean = 0

        for i in range(0, trainSize, trBatchSize):



            if ((batchID % 1000) == 0):
                print("batchID:"+str(batchID)+'/'+str(trainImage.size()[0]/trBatchSize) )

            if i+trBatchSize>=trainSize:
                input = trainImage[i:]
            else:
                input = trainImage[i:i+trBatchSize]
            
            
            #--Transforming Data
            input = ChexnetTrainer.trans_train(input, nnArchitecture)
            input = input.type(torch.cuda.FloatTensor)



            #print("----------------FEEDING BATCH TO MODEL-----------------")
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(copy.deepcopy(input))

            varInput = varInput.cuda()
            varTarget = varTarget.cuda()

           
        
            varOutput = model(varInput)
            lossvalue = loss(varOutput,varTarget).cuda()

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()


            losstensor = lossvalue.data[0]
            losstensorMean += losstensor
            
            lossTrain += losstensor.item()
            lossTrainNorm += 1

            batchID += 1

            # if batchID>2:
            #     break


        

     
        
        outLoss = lossTrain / lossTrainNorm
        print("Training loss : ", outLoss)

        print("varInput:", varInput[0])
        print("varOutput:",varOutput[0])

        # print("===================== Model training for this epoch finished====================================")
        
        return outLoss






    
    def computeAUROC (dataGT, dataPRED, classCount):


        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i], average = 'weighted'))
                
        return outAUROC

    


    def val (parentPath, model, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimestamp, threshold, device):   
            
            print("\n\n\n")
            print("Inside val funtion")
            CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            
            
            #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
            print("Is model==None:", model==None)
            print("Is pathModel==None:", pathModel==None)

            if pathModel!=None:
                model = ChexnetTrainer.loadModel(nnArchitecture, nnClassCount, nnIsTrained)
                # model = torch.nn.DataParallel(model)
                model.to(device)

                if os.path.isfile(pathModel):
                    print("=> loading checkpoint: ", pathModel)
                    modelCheckpoint = torch.load(pathModel)
                    model.load_state_dict(modelCheckpoint['state_dict'])
                    print("=> loaded checkpoint: ", pathModel)
                else:
                    print("=> no checkpoint found: ", pathModel)



            # model = torch.nn.DataParallel(model)
            # model.to(device)

            #---LOSS
            loss = torch.nn.BCELoss(size_average = True)
        


            print("\n============================ Loading data into RAM ======================================== ")

            if nnArchitecture == 'INCEPTION-V3':
                valImage =torch.load(parentPath+'/database/val_only_299.pth')
            else:
                valImage =torch.load(parentPath+'/database/val_only_224.pth')

            valLabel =torch.load(parentPath+'/database/labels_val_only_14.pth')
            valLabel = valLabel.type(torch.cuda.FloatTensor)

            outGT = torch.cuda.FloatTensor()
            outPRED = torch.cuda.FloatTensor()

            print("============================= Evaluation of model starts ====================================")
            model.eval()

            valSize = int(valImage.shape[0])
            corrects = torch.zeros((1,14)).cuda()
            batchID = 1
            counter = 0
            losstensorMean = 0 
            lossTrainNorm = 0
            auroc_list=[]

            with torch.no_grad():
                for i in range(0, valSize, trBatchSize):

                    if ((batchID%100)==0):
                        print("batchID:"+str(batchID)+'/'+str(valImage.size()[0]/trBatchSize) )

                    if i+trBatchSize>=valSize:
                        input = valImage[i:]
                        target = valLabel[i:]
                    else:
                        input = valImage[i:i+trBatchSize]
                        target = valLabel[i:i+trBatchSize]



                    outGT = torch.cat((outGT, target), 0)
                    bs, c, h, w = input.size()

                    input = ChexnetTrainer.trans_val(input, nnArchitecture)
                    input = input.type(torch.cuda.FloatTensor)
                    #input /= 255.0
                    

                    varInput = torch.autograd.Variable(input.view(-1, c, h, w))
                    varTarget = torch.autograd.Variable(target)
                    
                    out = model(varInput.cuda())
                
                    outMean = out.view(bs,-1)

                    lossvalue = loss(outMean,target).cuda()
                    losstensorMean += lossvalue.data[0].cpu()
                    
                    outPRED = torch.cat((outPRED, outMean.data), 0)

                    out_max = outMean.data>=threshold.data.float()
                    counter += ((out_max.sum(1) == 0)).sum()

                    out_max = torch.cuda.FloatTensor(out_max.float())
                    
                    corrects += ((out_max==target).sum(0)).float()

                    batchID+=1
                    lossTrainNorm+=1

                    # if batchID>200:
                    #     break
                    

                
               

            # print("\n--------------- Accuracy of prediction on the 14 classes ---------------------")
            # accuracy = corrects/((batchID-1)*trBatchSize)*100
            # for element in range(nnClassCount):
            #     print(CLASS_NAMES[element], ' ', accuracy[0][element])
            
            losstensorMean = losstensorMean/lossTrainNorm
            print('Number of cases of no disease found:', counter)
            
                   
            print("\n-------------- AUROC score of the 14 classes -------------------")
            aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED,nnClassCount)
            aurocMean = np.array(aurocIndividual).mean()
            
            print ('AUROC mean ', aurocMean)
            
            for i in range (0, len(aurocIndividual)):
                print (CLASS_NAMES[i], ' ', aurocIndividual[i])        
                
            print("Eval Loss:", (losstensorMean) )
         
            return aurocMean, losstensorMean



                
        #-------------------------------------------------------------------------------- 
        
















    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (parentPath, model, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimestamp, threshold, device):   
        
        print("\n\n\n")
        print("Inside test funtion")
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        #cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        print("Is model==None:", model==None)
        print("Is pathModel==None:", pathModel==None)

        if pathModel!=None:
            model = ChexnetTrainer.loadModel(nnArchitecture, nnClassCount, nnIsTrained)
            #model = torch.nn.DataParallel(model)
            model.to(device)

            if os.path.isfile(pathModel):
                print("=> loading checkpoint: ", pathModel)
                modelCheckpoint = torch.load(pathModel)
                model.load_state_dict(modelCheckpoint['state_dict'])
                print("=> loaded checkpoint: ", pathModel)
            else:
                print("=> no checkpoint found: ", pathModel)




        # model.to(device)
        
        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS

        # transformList = []
        # transformList.append(transforms.Resize(transResize))
        # transformList.append(transforms.TenCrop(transCrop))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # transformSequence=transforms.Compose(transformList)

        

        print("\n============================ Loading data into RAM ======================================== ")

        if nnArchitecture == 'INCEPTION-V3':
            testImage =torch.load(parentPath+'/database/test342.pth')
        else:
            testImage =torch.load(parentPath+'/database/test256.pth')

        testLabel =torch.load(parentPath+'/database/labelsTest224_14.pth')
        testLabel = testLabel.type(torch.cuda.FloatTensor)

        outGT = torch.cuda.FloatTensor()
        outPRED = torch.cuda.FloatTensor()

        print("============================= Evaluation of model starts ====================================")
        model.eval()

        testSize = 25596
        corrects = torch.zeros((1,14)).cuda()
        batchID = 1
        counter = 0
        auroc_list=[]

        with torch.no_grad():
            for i in range(0, testSize, trBatchSize):

                if ((batchID%100)==0):
                    print("batchID:"+str(batchID)+'/'+str(testImage.size()[0]/trBatchSize) )

                if i+trBatchSize>=testSize:
                    input = testImage[i:]
                    target = testLabel[i:]
                else:
                    input = testImage[i:i+trBatchSize]
                    target = testLabel[i:i+trBatchSize]



                outGT = torch.cat((outGT, target), 0)

                input = ChexnetTrainer.trans_test(input, nnArchitecture, transCrop)
                input = input.type(torch.cuda.FloatTensor)
                #input /= 255.0
                bs, n_crops, c, h, w = input.size()
                

                varInput = torch.autograd.Variable(input.view(-1, c, h, w))
                varTarget = torch.autograd.Variable(target)
                
                out = model(varInput.cuda())
            
                outMean = out.view(bs, n_crops, -1).mean(1)
                outPRED = torch.cat((outPRED, outMean.data), 0)

                out_max = outMean.data>=threshold.data.float()
                counter += ((out_max.sum(1) == 0)).sum()

                out_max = torch.cuda.FloatTensor(out_max.float())
                
                corrects += ((out_max==target).sum(0)).float()

                batchID+=1

                # if batchID>200:
                #     break
                

            
           

        # print("\n--------------- Accuracy of prediction on the 14 classes ---------------------")
        # accuracy = corrects/((batchID-1)*trBatchSize)*100
        # for element in range(nnClassCount):
        #     print(CLASS_NAMES[element], ' ', accuracy[0][element])
        
        
        print('Number of cases of no disease found:', counter)
        
               
        print("\n-------------- AUROC score of the 14 classes -------------------")
        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED,nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])        
            
     
        return aurocMean



            
    #-------------------------------------------------------------------------------- 
    

