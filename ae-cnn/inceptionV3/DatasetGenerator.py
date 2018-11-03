import os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('L')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)

        imageData = np.array(imageData)
        imageData = torch.from_numpy(imageData).view(1, imageData.shape[0], imageData.shape[1])
        # print("index:", index)
        # print("imageData shape:", imageData.shape)
        # print("imageData dtype", imageData.dtype)

        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
        
    def get(self, index):
            
            imagePath = self.listImagePaths[index]
            
            imageData = Image.open(imagePath).convert('L')
            imageLabel= torch.FloatTensor(self.listImageLabels[index])
            
            if self.transform != None: imageData = self.transform(imageData)

            imageData = np.array(imageData)
            imageData = torch.from_numpy(imageData).view(1, imageData.shape[0], imageData.shape[1])
            # print("index:", index)
            # print("imageData shape:", imageData.shape)
            # print("imageData dtype", imageData.dtype)

            
            return imageData, imageLabel
            
    #-------------------------------------------------------------------------------- 
    
    def length(self):
        
        return len(self.listImagePaths)





class DatasetGeneratorTest (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('L')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)

        imageData = np.array(imageData)
        # print("imageData shape", imageData.shape)
        imageData = torch.from_numpy(imageData)
        # imageData = torch.from_numpy(imageData).view(1, imageData.shape[0], imageData.shape[1])
        # print("index:", index)
        # print("imageData shape:", imageData.shape)
        # print("imageData dtype", imageData.dtype)

        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
        
    def get(self, index):
            
            imagePath = self.listImagePaths[index]
            
            imageData = Image.open(imagePath).convert('L')
            imageLabel= torch.FloatTensor(self.listImageLabels[index])
            
            if self.transform != None: imageData = self.transform(imageData)

            imageData = np.array(imageData)
            # print("imageData before shape:", imageData.shape)
            imageData = torch.from_numpy(imageData)#.view(imageData.shape[0], 1, imageData.shape[1], imageData.shape[2])
            # print("index:", index)
            # print("imageData after shape:", imageData.shape)
            # print("imageData dtype", imageData.dtype)

            
            return imageData, imageLabel
            
    #-------------------------------------------------------------------------------- 
    
    def length(self):
        
        return len(self.listImagePaths)