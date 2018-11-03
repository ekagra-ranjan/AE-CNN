import numpy as np
import torch
import torchvision

class lossFunc():

    
    def loss_func(yhat, y, device):
        
        m = yhat.size()[0]
        nnClasses = yhat.size()[1]
        epsilon = 1e-10
        
        class_weights = torch.Tensor([[ 0.9043,  0.9803,  0.8999,  0.8407,  0.9534,  0.9456,  0.9899,
          0.9695,  0.9670,  0.9841,  0.9836,  0.9855,  0.9741,  0.9984]]).to(device)
        
        pos_wt = (torch.ones((m, nnClasses), dtype=torch.float)).to(device)*class_weights.float()
        neg_wt = 1-pos_wt
        wt = (y)*(pos_wt) + (1-y)*(neg_wt)
        
        unweighted_loss = - (y*(torch.log(yhat+epsilon)) + (1-y)*(torch.log(1-yhat+epsilon)))
        weighted_loss = unweighted_loss*wt
        
        loss_per_class = ((weighted_loss.sum(0)).data.view(1,nnClasses)).data
        loss = (weighted_loss.sum())/m
        
        
        if False:
        
            print("log y:", torch.log(y))
            print("pos_wt", pos_wt)
            print("wt", wt)
            print("unweighted_loss", unweighted_loss)
            print("weighted_loss", weighted_loss)

            print("y grad", y.requires_grad)
            print("yhat grad", yhat.requires_grad)
            print("class wt grad", class_weights.requires_grad)
            print("pos_wt grad", pos_wt.requires_grad)
            print("neg_wt", neg_wt.requires_grad)
            print("wt grad", wt.requires_grad)
            print("unweighted grad:", unweighted_loss.requires_grad)
            print("weighted_loss", weighted_loss.requires_grad)
            print("loss grad", loss.requires_grad)

        
        
        return loss, loss_per_class
        
