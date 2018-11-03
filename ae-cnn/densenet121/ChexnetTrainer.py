import os
import numpy as np
import time
from time import time as now
import sys
import random
import copy
from sys import getsizeof

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

from PIL import Image,ImageFile
from matplotlib import pyplot as plt
from skimage import io

from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import average_precision_score as ap
from DensenetModels import AECNN, AECNN0, Resnet18
from DatasetGenerator import DatasetGenerator, DatasetGeneratorTest
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
    
    #---Shuffle when just pretraining AE: takes in patch, labels(not corresponding to patch) and returns shuffled patches and vanilla labels
    def shufflefinal(it, il, N):
        for loop in range(N//2):

            irand1 = np.random.randint(0, N)
            irand2 = np.random.randint(0, N)

            ivar = it[irand1].clone()
            it[irand1] = it[irand2]
            it[irand2] = ivar
            
            ivar = il[irand1].clone()
            il[irand1] = il[irand2]
            il[irand2] = ivar

        return it, il


    #----TRANFORMATION FUNCTION

    #takes 256x256 and returns 224x224 (cropping)
    def trans_train(x, nnArchitecture, transCrop):
        
        # if nnArchitecture != 'INCEPTION-V3':
        #     normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #normalize = transforms.Normalize([0.5057, 0.5057, 0.5057], [0.251, 0.251, 0.251])


        transformList = []
        transformList.append(transforms.ToPILImage())
        transformList.append(transforms.RandomCrop(transCrop) )
        # transformList.append(transforms.RandomRotation(5))
        transformList.append(transforms.RandomRotation(5))
        # transformList.append(transforms.ColorJitter(brightness=0, contrast=0.25))
        transformList.append(transforms.RandomHorizontalFlip())
        # transformList.append(transforms.RandomVerticalFlip())
        transformList.append(transforms.ToTensor())
        # if nnArchitecture != 'INCEPTION-V3':
        #     transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)
        
        # y = torch.zeros( size=(x.shape), dtype = torch.float32)
        y = torch.zeros( size=(x.shape[0], x.shape[1], transCrop, transCrop), dtype = torch.float32)
        
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
        # transformList.append(transforms.Resize(transCrop))
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


    # -----ChexnetTrainer.store() - function to store the output of encoder of AE-CNN. This can be used for visualisation of latent code produced by encoder
    def store(parentPath, pathModel, nnArchitecture, nnClassCount, trBatchSize, device):

        model = AECNN0(nnClassCount)
        model.to(device)


        #--LOAD MODEL
        if pathModel != None:
            modelCheckpoint = torch.load(pathModel, map_location=lambda storage, loc: storage)
            model.load_state_dict(modelCheckpoint['state_dict'])
            print("LOADED FROM PATHMODEL:", pathModel)
        else:
            print("PATHMODEL could not be loaded:", pathModel)

        #train size: 76524, val size 10000, test size: 25596
        # output_encoding = torch.ByteTensor(10000, 3, 224, 224)
        output_encoding = torch.ByteTensor(25596, 3, 224, 224)



        ImageFile.LOAD_TRUNCATED_IMAGES = True


        # file_path = '/home/deepsip/ChestX/dataset/val_only_15.txt'
        file_path = '/home/deepsip/ChestX/dataset/test_final_15.txt'


        file_ptr = open(file_path,'r')
        batchID = 0
        basewidth = 896

        for line in file_ptr:

            if batchID%1000==0:
                print("batchID:", batchID)
            
            img_name = line.split(" ")[0]
            # path="/home/deepsip/ChestX/database/train_set/"+img_name
            path="/home/deepsip/ChestX/database/test_set/"+img_name
            
            img = Image.open(path)
            img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
            img=img.convert('L')

            np_image=np.array(img)
            image_tensor=torch.from_numpy(np_image)

            input = image_tensor.float()
            input/=255
            input = input.view(1, 1, input.shape[0], input.shape[1])

            varInput = torch.autograd.Variable(input)
            varInput = varInput.to(device)


            varOutput = model.encoder(varInput)
            #print(varOutput.shape)
            output_img = (varOutput*255).byte()
            # print(type(output_img))
            # print(output_img.dtype)

            output_encoding[batchID] = output_img

            batchID+=1

            # if batchID==10:
            #     break
            

        print("No of images encoded:", batchID)

        # torch.save(output_encoding,"/home/deepsip/ChestX/database/val_only_224_autoencoder.pth")
        torch.save(output_encoding,"/home/deepsip/ChestX/database/test224_autoencoder.pth")





    
    def train (parentPath, pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, threshold , device, checkpoint1, checkpoint2, checkpoint3):

        print("Inside train function:")
    
        #-------------------- SETTINGS: NETWORK ARCHITECTURE

        
        # -----AECNN  - class for InceptionV3 type model models which need upscaling of encoder output (224x224) before passing to classifier requiring 299x299
        # -----AECNN0 - class for resnets and densenets type models requiring 224x224 images 
        model  = AECNN0(nnClassCount)
        # model  = AECNN(nnClassCount)

        #model = Resnet18(nnClassCount, True)
        #--------GPU
        #model = torch.nn.DataParallel(model)
        #model0.to(device)
        model.to(device)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 2, mode = 'min')
                
        

        #---- Load checkpoint 
        #---checkpoint1 is for all
        epochStart = 0
        

        if checkpoint3 != None:
            modelCheckpoint = torch.load(checkpoint3, map_location=lambda storage, loc: storage)
            model.load_state_dict(modelCheckpoint['state_dict'])
            # model.encoder.load_state_dict(modelCheckpoint['state_dict'], strict=False)
            # model.decoder.load_state_dict(modelCheckpoint['state_dict'], strict=False)
            
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            # optimizer=optimizer.cpu()
            epochStart = modelCheckpoint['epoch']
            #optimizer.param_groups[0]['lr'] = 1e-5
            print("LOADED FROM CHECKPOINT:", checkpoint3)

        else:

            if checkpoint1 != None:
                modelCheckpoint = torch.load('encoder.pth.tar', map_location=lambda storage, loc: storage)
                model.encoder.load_state_dict(modelCheckpoint['state_dict'])
                print("encoder loaded")
                modelCheckpoint = torch.load('decoder.pth.tar', map_location=lambda storage, loc: storage)
                model.decoder.load_state_dict(modelCheckpoint['state_dict'])
                print("decoder loaded")
                                

                #modelCheckpoint = torch.load(checkpoint1, map_location=lambda storage, loc: storage)
                #model0.load_state_dict(modelCheckpoint['state_dict'])
                #print("checkpoint1 loaded, noe ladoing encoder decoder")
                #model.enoder = model0.encoder
                #model.decoder = model0.decoder
                # model.encoder.load_state_dict(modelCheckpoint['state_dict'], strict=False)
                # model.decoder.load_state_dict(modelCheckpoint['state_dict'], strict=False)
                
                # optimizer.load_state_dict(modelCheckpoint['optimizer'])
                # epochStart = modelCheckpoint['epoch']
                #optimizer.param_groups[0]['lr'] = 1e-5
                print("LOADED FROM CHECKPOINT:", checkpoint1)

            #---checkpoint is only for classifier d121
            if checkpoint2 != None:
                modelCheckpoint = torch.load(checkpoint2, map_location=lambda storage, loc: storage)
                # print("modelCheckpoint:\n", modelCheckpoint['state_dict'])
                # print("model.classifier", model.classifier)
                model.classifier.load_state_dict(modelCheckpoint['state_dict'],strict = False)
                # optimizer.load_state_dict(modelCheckpoint['optimizer'])
                # epochStart = modelCheckpoint['epoch']
                #optimizer.param_groups[0]['lr'] = 1e-5
                print("LOADED FROM CHECKPOINT:", checkpoint2)

        # model = torch.nn.DataParallel(model)
        # model.to(device)


        #-------------------- SETTINGS: DATA TRANSFORMS
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        # transformList.append(transforms.RandomResizedCrop(transCrop))
        # transformList.append(transforms.Resize(896))
        transformList.append(transforms.RandomCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        #transformList.append(transforms.ToTensor())
        # transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)



        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=2, pin_memory=False)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=2,  pin_memory=False)

        
    
        #--------------------RAM DATA
        # print("============================ Loading data into RAM ======================================== ")

        
        # trainImage = torch.load(parentPath+"/database/autoencoder_32_1948_128.pth")
        # trainLabel =torch.load(parentPath+"/database/labels_train_only_"+str(nnClassCount)+".pth")
        # trainLabel = trainLabel.type(torch.cuda.FloatTensor)
     
        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000    
        #print("Total No of batches : " + str(trainImage.shape[0]/trBatchSize))
        auc_per_5_epochs = []
        auc_per_epoch = []
        loss_train_per_epoch = []
        loss_val_per_epoch = []

        print('\n')
        
        for epochID in range (epochStart, trMaxEpoch):

            if (epochID+1)%5==0 and epochID!=0:
                optimizer.param_groups[0]['lr'] /= 10

            print("\nEpoch: ", epochID+1)
            print("LR: ", optimizer.param_groups[0]['lr'])
            
            # shuffled_img, shuffled_label = ChexnetTrainer.shufflefinal1(trainImage, trainLabel)
            # print("===============================Shuffling done==================================")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            
        #----------------------------------------------------------------training             
            tic = now()
            lossTrain = ChexnetTrainer.trainEpoch(model, nnArchitecture, datasetTrain, dataLoaderTrain, optimizer, trBatchSize, transCrop, nnClassCount, threshold, epochID, launchTimestamp, device) 
            
            loss_train_per_epoch.append(lossTrain)
            
            torch.save(loss_train_per_epoch, './loss/' + nnArchitecture+'-'+str(trMaxEpoch)+'-'+'loss_train_mean-'+str(launchTimestamp) )
            # print("Saved Loss for this epoch")
            
            toc = now()

            print("Time taken in this epoch {}".format(toc-tic))
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
        
            
            if ((epochID + 1)%5 != 7):

                print ('\nValidating the trained model ')
                aurocMean, losstensor = ChexnetTrainer.val(parentPath, datasetVal, dataLoaderVal, model, None, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimestamp, threshold, device)
                # losstensor = ChexnetTrainer.val(parentPath, datasetVal, dataLoaderVal, model, None, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimestamp, threshold, device)
            
                # auc_per_5_epochs.append(aurocMean)
                loss_val_per_epoch.append(losstensor)
                torch.save(auc_per_5_epochs, './AUC/' + nnArchitecture+'-'+str(trMaxEpoch)+'-'+'auc_mean-'+str(launchTimestamp))
                torch.save(loss_val_per_epoch, './loss/' + nnArchitecture+'-'+str(trMaxEpoch)+'-'+'loss_val_mean-'+str(launchTimestamp))
                print("AUC saved\nSaving Model")

                #---STEP SCHEDULER
                # scheduler.step(losstensor.data[0])
        

                lossMIN = lossTrain    
                # torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + nnArchitecture + '-' + 'epoch-' + str(epochID+1) + '-loss-' + str(lossTrain) + '-' + str(launchTimestamp) + '.pth.tar')
                # latestModelName = 'm-' + nnArchitecture + '-' + 'epoch-' + str(epochID+1) + '-loss-' + str(lossTrain) + '-' + str(launchTimestamp) + '.pth.tar'
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, './models/' + 'm-' + nnArchitecture + '-' + 'epoch-' + str(epochID+1) + '-auc-' + str(aurocMean) + '-' + str(launchTimestamp) + '.pth.tar')
                latestModelName = 'm-' + nnArchitecture + '-' + 'epoch-' + str(epochID+1) + '-auc-' + str(aurocMean) + '-' + str(launchTimestamp) + '.pth.tar'
            
                print(latestModelName, " Saved")
                
                
        
        return latestModelName
                     
    #-------------------------------------------------------------------------------- 
       
        

    #--------------------------------------------------CUSTOM TRAINING WITH CUSTOM LOSS
    def trainEpoch(model, nnArchitecture, datasetTrain, dataLoader, optimizer, trBatchSize, transCrop, classCount, threshold, epochID, launchTimestamp, device):

        #model.train()
        # t=now()
       
        #trainSize = trainImage.shape[0]
        trainSize = 76524

        #loss = torch.nn.MSELoss(size_average = True)
        loss1 = torch.nn.MSELoss(size_average = True)
        loss2 = torch.nn.BCELoss(size_average = True)
        # print('================= Model Training starts for this epoch ===============================')

        batchID = 1

        lossTrain = 0
        lossTrain1 = 0
        lossTrain2 = 0
        lossTrainNorm = 0        
        losstensorMean = 0
        losstensorMean1 = 0
        losstensorMean2 = 0

        # input = torch.ByteTensor(trBatchSize, 1, transCrop, transCrop)
        # target = torch.FloatTensor(trBatchSize, classCount)

        # for i in range(0, trainSize, trBatchSize):
        print("Inside Train Epoch")
        
        random_index = random.sample(range(0, trainSize), trainSize)
        print("No of train batches:", trainSize/trBatchSize)
        # for i in range(0, trainSize, trBatchSize):




        # trBatchSize = 5
        pth_order = random.sample(range(0, 75000, 10000), 8)
        #pth_order = [0]
        
        for pth_name in pth_order:
            print("loading", str(pth_name)+'.pth')
            trainImage, trainLabel = torch.load('../dataset/train_1024/'+str(pth_name)+'.pth')
            print("trainImage shape", trainImage.shape)
            print("trainLabel shape", trainLabel.shape)
            print("# of img:", trainLabel.shape[0])
            trainSizePth = trainLabel.shape[0]
            print("shuffling")
            trainImage, trainLabel = ChexnetTrainer.shufflefinal(trainImage, trainLabel, trainSizePth)
            #10000x1x896x896




            for i in range(0, trainSizePth, trBatchSize):



                if ((batchID % 1000) == 0):
                    print("batchID:"+str(batchID)+'/'+str(trainSize/trBatchSize) )
                    torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, 'm-' + nnArchitecture + '-' + 'epoch-' + str(epochID+1) +  '-' + str(launchTimestamp) + '.pth.tar')
                

                if i+trBatchSize>=trainSizePth:
                    input = trainImage[i:]
                    target = trainLabel[i:]
                else:
                    input = trainImage[i:i+trBatchSize]
                    target = trainLabel[i:i+trBatchSize]
                
                #input has dim bsx1x896x896
                # print("input shape before view", input.shape)

                #--Transforming Data

                # print("input shape after view", input.shape)
                input = ChexnetTrainer.trans_train(input, nnArchitecture, transCrop)
                input = input.type(torch.cuda.FloatTensor)
                # print("input shape after trans", input.shape)
                #input /= 255.0


                
                  
                #print("----------------FEEDING BATCH TO MODEL-----------------")
                varInput = torch.autograd.Variable(input).to(device)
                varTarget1 = input.to(device)
                varTarget2 = torch.autograd.Variable(target).to(device)


                # varInput dim: bsx1x896x896
                varOutput1, varOutput2 = model(varInput)
                lossvalue1 = loss1(varOutput1, varTarget1)
                lossvalue2 = loss2(varOutput2, varTarget2)
                lossvalue = 0.1*lossvalue1 + 0.9*lossvalue2 # weighting bw MSE and BCE resp.

                
                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()


                losstensor = lossvalue.data[0]
                losstensorMean += losstensor

                losstensor1 = lossvalue1.data[0]
                losstensorMean1 += losstensor1
                
                losstensor2 = lossvalue2.data[0]
                losstensorMean2 += losstensor2
                
                lossTrain += losstensor.item()
                lossTrain1 += losstensor1.item()
                lossTrain2 += losstensor2.item()
                lossTrainNorm += 1

                
                if batchID%1000==0:
                    print("lossTrain : ", lossTrain/lossTrainNorm)
                    print("lossTrain1: ", lossTrain1/lossTrainNorm)
                    print("lossTrain2: ", lossTrain2/lossTrainNorm)

                batchID += 1

                #if batchID>200:
                #    break

                # print("lossvalue:", lossvalue)    
                # print("lossvalue1:", lossvalue1)
                # print("lossvalue2:", lossvalue2)

                # print("time taken for", batchID+1, " :", (now()-t)/(batchID+1) )



            del trainImage
            del trainLabel

     
       

        outLoss = lossTrain / lossTrainNorm
        outLoss1 = lossTrain1 / lossTrainNorm
        outLoss2 = lossTrain2 / lossTrainNorm
        
        print("Training loss : ", outLoss)
        print("Training loss 1: ", outLoss1)
        print("Training loss 2: ", outLoss2)


        # print("varInput:", varInput[0])
        # print("varOutput1:",varOutput1[0])

        # print("===================== Model training for this epoch finished====================================")
        
        return outLoss







    # ----- function to compute AUROC, AUPRC, AP
    def computeAUROC (dataGT, dataPRED, classCount):


        outAUROC = []
        outAUPRC = []
        outAP = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i], average = 'weighted'))
            outP, outR, _ = prc(datanpGT[:, i], datanpPRED[:, i])
            outAUPRC.append(auc(outR, outP, False))
            outAP.append(ap(datanpGT[:, i], datanpPRED[:, i]))
                
        return outAUROC, outAUPRC, outAP

    


    def val (parentPath, datasetVal, dataLoader, model, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimestamp, threshold, device):   
            
            # print("\n\n\n")
            print("\nInside val funtion")
            CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            
            
            #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
            print("Is model==None:", model==None)
            print("Is pathModel==None:", pathModel==None)

            #While using val after training, pathModel is set None as model is directly passed as arg 
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
            #loss = torch.nn.BCELoss(size_average = True)
            loss1 = torch.nn.MSELoss(size_average = True)
            loss2 = torch.nn.BCELoss(size_average = True)
        


            # print("\n============================ Loading data into RAM ======================================== ")

            # if nnArchitecture == 'INCEPTION-V3':
            #     valImage =torch.load(parentPath+'/database/val_only_299.pth')
            # else:
            #     valImage =torch.load(parentPath+'/database/val_only_224.pth')

            # valLabel =torch.load(parentPath+'/database/labels_val_only_14.pth')
            # valLabel = valLabel.type(torch.cuda.FloatTensor)

            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
            

            print("============================= Evaluation of model starts ====================================")
            model.eval()

            #valSize = int(valImage.shape[0])
            #corrects = torch.zeros((1,14)).cuda()
            batchID = 1
            #counter = 0
            losstensorMean = 0.0 
            losstensorMean1 = 0.0 
            losstensorMean2 = 0.0 
            lossTrainNorm = 0.0
            auroc_list=[]

            # input = torch.ByteTensor(trBatchSize, 1, 896, 896)
            # target = torch.FloatTensor(trBatchSize, nnClassCount)


            valSize=10000
            pth_order = random.sample(range(0, valSize, 10000), 1)
            # pth_order = [0]
            
            for pth_name in pth_order:
                print("loading", str(pth_name)+'.pth')
                valImage, valLabel = torch.load('../dataset/val/'+str(pth_name)+'.pth')
                # print("valImage shape", valImage.shape)
                # print("valLabel shape", valLabel.shape)
                print("# of img:", valLabel.shape[0])
                valSizePth = valLabel.shape[0]
                #10000x1x896x896


                with torch.no_grad():
                    for i in range(0, valSizePth, trBatchSize):


                        if ((batchID%100)==0):
                            print("batchID:"+str(batchID)+'/'+str(valSize/trBatchSize) )

                        if i+trBatchSize>=valSize:
                            input = valImage[i:]
                            target = valLabel[i:]
                        else:
                            input = valImage[i:i+trBatchSize]
                            target = valLabel[i:i+trBatchSize]
                        
                        # input = input.cuda()
                        # target = target.cuda()
                        #input has dim bsx1x896x896
                        # print("input shape before view", input.shape)

                        #--Transforming Data
                        # input = input.view(-1, 3, 224, 224)

                        outGT = torch.cat((outGT, target), 0)
                        bs, c, h, w = input.size()
                        # print("input shape after view", input.shape)
                        input = ChexnetTrainer.trans_val(input, nnArchitecture)
                        input = input.type(torch.cuda.FloatTensor)
                        # print("input shape after trans", input.shape)
                        #input /= 255.0


                        # varInput = torch.autograd.Variable(input.view(-1, c, h, w))
                        varInput = torch.autograd.Variable(input).to(device)
                        varTarget1 = input.to(device)
                        varTarget2 = torch.autograd.Variable(target).to(device)

                        varOutput1, varOutput2 = model(varInput)
                        varOutput2Mean = varOutput2.view(bs,-1)
                        
                        lossvalue1 = loss1(varOutput1, varTarget1)
                        lossvalue2 = loss2(varOutput2Mean, varTarget2)
                        lossvalue =  lossvalue1 + lossvalue2

                        losstensorMean += lossvalue.data[0].cpu()
                        losstensorMean1 += lossvalue1.data[0].cpu()
                        losstensorMean2 += lossvalue2.data[0].cpu()

                        # print("lossvalue1:", lossvalue1)
                        # print("lossvalue:", lossvalue)
                        # print("lossvalue1:", lossvalue1)
                        # print("lossvalue2:", lossvalue2)
                        # print("lossvalue1+lossvalue2:", lossvalue1+lossvalue2)
                        # print("\n")
                        
                        outPRED = torch.cat((outPRED, varOutput2Mean.data.cpu()), 0)

                        lossTrainNorm+=1

                        batchID+=1

                        # if batchID>200:
                        #     break
                        
                        
                del valImage
                del valLabel       



            losstensorMean = losstensorMean/lossTrainNorm
            losstensorMean1 = losstensorMean1/lossTrainNorm
            losstensorMean2 = losstensorMean2/lossTrainNorm


            print("\n-------------- AUC scores of the "+str(nnClassCount)+" classes -------------------")
            aurocIndividual, auprcIndividual, apIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
            aurocMean = np.array(aurocIndividual).mean()
            auprcMean = np.array(auprcIndividual).mean()
            apMean = np.array(apIndividual).mean()
            
            print ('AUROC mean ', aurocMean)
            
            for i in range (0, len(aurocIndividual)):
                print (CLASS_NAMES[i], ' ', aurocIndividual[i])

            print('\n\nAUPRC mean', auprcMean)
            for i in range(0, len(auprcIndividual)):
                print(CLASS_NAMES[i], ' ', auprcIndividual[i])

            print("\n\nAP mean", apMean)

            for i in range(0, len(apIndividual)):
                print(CLASS_NAMES[i], ' ', apIndividual[i])
            
         
              
            print("Eval Loss:", (losstensorMean) )
            print("lossvalue1:", losstensorMean1)
            print("lossvalue2:", losstensorMean2)
         
            return aurocMean, losstensorMean
            # return losstensorMean



                
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
    
    def test (parentPath, model, pathModel, pathDirDataTest, pathFileTest, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimestamp, threshold, device):   
        
        #---TRANSFORMATION
        transformList = []
        # transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.FiveCrop(transCrop))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([ crop for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        num_crops = 5

        #---DATASETGENERTOR
        datasetTest = DatasetGeneratorTest(pathImageDirectory=pathDirDataTest, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=False)
        

        print("\n\n\n")
        print("Inside test funtion")
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        #cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        print("Is model==None:", model==None)
        print("Is pathModel==None:", pathModel==None)

        if pathModel!=None:
            model = AECNN0(nnClassCount)
            # model = torch.nn.DataParallel(model)
            model.to(device)

            if os.path.isfile(pathModel):
                print("=> loading checkpoint: ", pathModel)
                modelCheckpoint = torch.load(pathModel)
                model.load_state_dict(modelCheckpoint['state_dict'])
                print("=> loaded checkpoint: ", pathModel)
            else:
                print("=> no checkpoint found: ", pathModel)




        
        loss1 = torch.nn.MSELoss(size_average = True)
        loss2 = torch.nn.BCELoss(size_average = True)



        outGT = torch.FloatTensor()
        outPRED = torch.FloatTensor()

        print("============================= Evaluation of model starts ====================================")
        model.eval()

        testSize = 25596
        # corrects = torch.zeros((1,14)).cuda()
        batchID = 1
        counter = 0
        losstensorMean = 0 
        losstensorMean1 = 0 
        losstensorMean2 = 0 
        lossTrainNorm = 0
        auroc_list=[]

        
        # print("Using number of crops:", num_crops)


        # input = torch.ByteTensor(trBatchSize*num_crops, 1, transCrop, transCrop)
        # target = torch.FloatTensor(trBatchSize, nnClassCount)



        # trBatchSize = 5
        pth_order = random.sample(range(0, testSize, 10000), 3)
        # pth_order = [0]
        
        for pth_name in pth_order:
            print("loading", str(pth_name)+'.pth')
            testImage, testLabel = torch.load('../dataset/test_1024/'+str(pth_name)+'.pth')
            # print("valImage shape", valImage.shape)
            # print("valLabel shape", valLabel.shape)
            print("# of img:", testLabel.shape[0])
            testSizePth = testLabel.shape[0]
            print("shuffling")
            # valImage, valLabel = ChexnetTrainer.shufflefinal(trainImage, trainLabel, trainSizePth)
            #5000x14x3x224x224


            with torch.no_grad():
                for i in range(0, testSizePth, trBatchSize):



                    if ((batchID%100)==0):
                        print("batchID:"+str(batchID)+'/'+str(testSize/trBatchSize) )

                    if i+trBatchSize>=testSize:
                        input = testImage[i:]
                        target = testLabel[i:]
                    else:
                        input = testImage[i:i+trBatchSize]
                        target = testLabel[i:i+trBatchSize]
                    
                    # input = input.cuda()
                    # target = target.cuda()
                    #input has dim bsx1x896x896
                    # print("input shape before view", input.shape)

                    #--Transforming Data
                    # input = input.view(-1, 3, 224, 224)

                    outGT = torch.cat((outGT, target), 0)
                    bs, c, h, w = input.size()
                    # print("input shape after view", input.shape)
                    input = ChexnetTrainer.trans_test(input, nnArchitecture, transCrop)
                    # print("input shape after trans", input.shape)
                    bs, n_crops, c, h, w = input.size()
                    # print("input shape after trans", input.shape)
                    #input /= 255.0
                    varInput = torch.autograd.Variable(input.view(-1, c, h, w)).to(device)
                

                    varTarget1 = input.view(-1, c, h, w).to(device)
                    varTarget2 = torch.autograd.Variable(target).to(device)

                    varOutput1, varOutput2 = model(varInput)
                    # print("varOutput2 shape:", varOutput2.shape)
                    # print("varOutput2:", varOutput2)
                    varOutput2Mean = varOutput2.view(bs,n_crops, -1).mean(1)
                    # varOutput2Mean = varOutput2.view(bs,-1)
                    # print("varOutput2.view()", varOutput2.view(bs,n_crops, -1))
                    # print("varOutput2 shape:", varOutput2.shape)

                    # print("i:", i)
                    # print("target:", target)
                    # print("varOutput2Mean:", varOutput2Mean)
                    
                    lossvalue1 = loss1(varOutput1, varTarget1)
                    lossvalue2 = loss2(varOutput2Mean, varTarget2)
                    lossvalue =  lossvalue1 + lossvalue2

                    losstensorMean += lossvalue.data[0].cpu()
                    losstensorMean1 += lossvalue1.data[0].cpu()
                    losstensorMean2 += lossvalue2.data[0].cpu()

                    
                    outPRED = torch.cat((outPRED, varOutput2Mean.data.cpu()), 0)
                    
                    lossTrainNorm+=1

                    batchID+=1
                    # if batchID>500:
                    #     break
                    
                    # print("lossvalue1:", lossvalue1)
                    # print("lossvalue:", lossvalue)
                    # print("lossvalue1:", lossvalue1)
                    # print("lossvalue2:", lossvalue2)
                    # print("lossvalue1+lossvalue2:", lossvalue1+lossvalue2)
                    # print("\n")



                    
            del testImage
            del testLabel       



            
           

        
        losstensorMean = losstensorMean/lossTrainNorm
        losstensorMean1 = losstensorMean1/lossTrainNorm
        losstensorMean2 = losstensorMean2/lossTrainNorm

     
               
        print("\n-------------- AUC scores of the 14 classes -------------------")
        aurocIndividual, auprcIndividual, apIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED,nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        auprcMean = np.array(auprcIndividual).mean()
        apMean = np.array(apIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])

        print('\n\nAUPRC mean', auprcMean)
        for i in range(0, len(auprcIndividual)):
            print(CLASS_NAMES[i], ' ', auprcIndividual[i])

        print("\n\nAP mean", apMean)

        for i in range(0, len(apIndividual)):
            print(CLASS_NAMES[i], ' ', apIndividual[i])
        
     
        return aurocMean



            
    #-------------------------------------------------------------------------------- 
    

