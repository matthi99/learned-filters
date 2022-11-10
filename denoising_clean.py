# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:22:41 2022

@author: Schwab Matthias
"""

import numpy as np
import torch
#import pywt
import ptwt


from matplotlib import pyplot as plt

import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader, TensorDataset

import os

import cv2


N_epochs=20
wave= 'haar'


folder = "data/faces/"
filelist = [d for d in os.listdir(folder)]
#filelist=filelist[0:n_train]

images=[]

for file in filelist:
    x=cv2.imread(folder+file,cv2.IMREAD_GRAYSCALE)
    x=x/255
    x=x-(1/2)
    x=2*x
    x=cv2.resize(x, (256,256))
    images.append(x)
    

images=np.reshape(images, (len(images), 256,256))
    
images=torch.from_numpy(images.astype(np.float32))

dataset=TensorDataset(images,images)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class learned_filter(torch.nn.Module):

    def __init__(self, indim, outdim):
        super().__init__()  
        self.lin1 = nn.Linear(in_features=indim, out_features=20)
        self.lin2 = nn.Linear(in_features=20, out_features=40)
        self.lin3 = nn.Linear(in_features=40, out_features=20)
        self.lin4 = nn.Linear(in_features=20, out_features=20)
        self.lin5 = nn.Linear(in_features=20, out_features=outdim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()


    def forward(self, inp):
        xx = self.actr(self.lin1(inp))
        xx = self.actr(self.lin2(xx))
        xx = self.actr(self.lin3(xx))
        xx = self.act1(self.lin4(xx))
        out = self.lin5(xx)
        return  out
    



net0=learned_filter(1,1).to(device)
net1=learned_filter(3,3).to(device)
net2=learned_filter(3,3).to(device)




LR=1e-4
net_optimizer = torch.optim.Adam(list(net0.parameters())+list(net1.parameters())+list(net2.parameters()), lr=LR)
sigma=0.1

for epoch in range(N_epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        count=0
        l=0
        oracle_loss=0
        for image, _ in dataloader:
            net_optimizer.zero_grad()
            image=image.to(device)
            noisy_image=image+sigma*torch.randn_like(image)
            coefficients= ptwt.wavedec2(noisy_image, wave, level=2)
            coef_level0=torch.flatten(coefficients[0])
            coef_level1=torch.stack((torch.flatten(coefficients[1][0]),torch.flatten(coefficients[1][1]), torch.flatten(coefficients[1][2])),-1)
            coef_level2=torch.stack((torch.flatten(coefficients[2][0]),torch.flatten(coefficients[2][1]), torch.flatten(coefficients[2][2])),-1)
            denoised0=net0(coef_level0[:,None])
            denoised1=net1(coef_level1)
            denoised2=net2(coef_level2)
            denoised_ges=[torch.reshape(denoised0,coefficients[0].shape), 
                          [torch.reshape(denoised1[:,0],coefficients[1][0].shape), torch.reshape(denoised1[:,1],coefficients[1][1].shape), torch.reshape(denoised1[:,2],coefficients[1][2].shape)],
                          [torch.reshape(denoised2[:,0],coefficients[2][0].shape), torch.reshape(denoised2[:,1],coefficients[2][1].shape), torch.reshape(denoised2[:,2],coefficients[2][2].shape)]]
            
            
            rec=ptwt.waverec2(denoised_ges, wave)
            
            loss=torch.mean((rec[0,...]-image)**2)
            loss.backward()
            net_optimizer.step()
            l+=loss.item()
            count+=1
        print(f"{'Loss'}={l/count}")
        
        
x=torch.linspace(-4, 4, 100, device=device)
x=x[:,None]
y=net0(x)

      

x=torch.linspace(-4, 4, 100, device=device)
x=torch.stack((x,x,x),-1)
y=net2(x)

plt.figure(2)
plt.plot(x[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy())
plt.plot(x[:,1].cpu().detach().numpy(),y[:,1].cpu().detach().numpy())
plt.plot(x[:,1].cpu().detach().numpy(),y[:,2].cpu().detach().numpy())
plt.plot(x[:,0].cpu().detach().numpy(),x[:,0].cpu().detach().numpy()) 
            

#           coeff=coeff.to(device)
#             coeff_reg=net(coeff)
#             a, [av, ah, ad], [v,h,d]=reshape(coeff_reg, L1, L2)
#             rec=ptwt.waverec2([a, [av, ah, ad], [v,h,d]], wave)
            
#             loss=torch.mean((rec-image)**2)
#             loss.backward()
#             net_optimizer.step()
            
#             l+=loss.item()
#             count+=1
#             #print(sigma)
#         print(f"{'Loss'}={l/count}")

# ptwt.wavedec2(image, wave, 