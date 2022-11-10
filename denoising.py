# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 11:09:24 2022

@author: matth
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
#from mpl_toolkits.axes_grid1 import ImageGrid



s2n_ratios=np.array([1,2,4,8])
N_epochs=20
wave= 'sym10'
n_train=100

folder = "data/faces/"
filelist = [d for d in os.listdir(folder)]
filelist=filelist[0:n_train]

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
    





net=learned_filter(2,1).to(device)
net_params = net.parameters()

LR=1e-4
net_optimizer = torch.optim.Adam(net_params, lr=LR)

def preprocess(image, wave):
    "richtig schlecht programmiert und viel zu langsam --> verbessern!!"
    a,[av,ah,ad], [v,h,d]=(ptwt.wavedec2(image, wave, level=2)) 
    
    L1=np.prod(a.shape)
    a=torch.reshape(a, (1,L1))
    
    av=torch.reshape(av,(1,L1))
    ah=torch.reshape(ah,(1,L1))
    ad=torch.reshape(ad,(1,L1))
    
    L2=np.prod(v.shape)
    v=torch.reshape(v,(1,L2))
    h=torch.reshape(h,(1,L2))
    d=torch.reshape(d,(1,L2))
    L=4*L1+3*L2
    
    temp=torch.zeros((L,2))
    for i in range(L1):
        temp[i,0]=1/3
        temp[i,1]=a[0,i]
    for i in range(L1):
        temp[L1+i,0]=2/3
        temp[L1+i,1]=av[0,i]
    for i in range(L1):
        temp[2*L1+i,0]=2/3
        temp[2*L1+i,1]=ah[0,i]
    for i in range(L1):
        temp[3*L1+i,0]=2/3
        temp[3*L1+i,1]=ad[0,i]
    for i in range(L2):
        temp[4*L1+i,0]=1
        temp[4*L1+i,1]=v[0,i]
    for i in range(L2):
        temp[4*L1+L2+i,0]=1
        temp[4*L1+L2+i,1]=h[0,i]
    for i in range(L2):
        temp[4*L1+2*L2+i,0]=1
        temp[4*L1+2*L2+i,1]=d[0,i]
    return temp, L1, L2

def reshape(coeff_reg, L1, L2):
    s1=int(np.sqrt(L1))
    s2=int(np.sqrt(L2))
    a=coeff_reg[0:L1,0]
    a=torch.reshape(a,(s1,s1))
    a=a[None, None, :, :]
    
    av=coeff_reg[L1:2*L1,0]
    av=torch.reshape(av,(s1,s1))
    av=av[None, None, :, :]
    
    ah=coeff_reg[2*L1:3*L1,0]
    ah=torch.reshape(ah,(s1,s1))
    ah=ah[None, None, :, :]
        
    ad=coeff_reg[3*L1:4*L1,0]
    ad=torch.reshape(ad,(s1,s1))
    ad=ad[None, None, :, :]
        
    v=coeff_reg[4*L1:4*L1+L2,0]
    v=torch.reshape(v,(s2,s2))
    v=v[None, None, :, :]
        
    h=coeff_reg[4*L1+L2:4*L1+2*L2,0]
    h=torch.reshape(h,(s2,s2))
    h=h[None, None, :, :]
        
    d=coeff_reg[4*L1+2*L2:4*L1+3*L2,0]
    d=torch.reshape(d,(s2,s2))
    d=d[None, None, :, :]
    return  a, [av, ah, ad], [v,h,d]

size = len(dataloader.dataset)


for k in range(len(s2n_ratios)):
    s2n_ratio=s2n_ratios[k]
    for epoch in range(N_epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        count=0
        l=0
        for image, _ in dataloader:
            net_optimizer.zero_grad()
            image=image.to(device)
            sigma=torch.mean(image**2)/s2n_ratio
            noisy_image=image+sigma*torch.randn_like(image)
            coeff, L1, L2=preprocess(noisy_image, wave)
            coeff=coeff.to(device)
            coeff_reg=net(coeff)
            a, [av, ah, ad], [v,h,d]=reshape(coeff_reg, L1, L2)
            rec=ptwt.waverec2([a, [av, ah, ad], [v,h,d]], wave)
            
            loss=torch.mean((rec-image)**2)
            loss.backward()
            net_optimizer.step()
            
            l+=loss.item()
            count+=1
            #print(sigma)
        print(f"{'Loss'}={l/count}")
        
    

    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(image[0,:,:].cpu().detach().numpy(), cmap='gray')
    
    plt.subplot(3,1,2)
    plt.imshow(noisy_image[0].cpu().detach().numpy(), cmap='gray')

    
    plt.subplot(3,1,3)
    plt.imshow(rec[0,0,:,:].cpu().detach().numpy(), cmap='gray')
    plt.savefig("plots/s2nr_"+str(s2n_ratio)+"/image.png",dpi=300, bbox_inches="tight", pad_inches=0.1, )
    plt.show()
    
    lower=-6
    upper=6
    x=np.zeros(200)
    for i in range(200):
        x[i]=lower+i*((upper-lower)/200)
    
        
    
    test=torch.zeros((3,200,2))
    for i in range(200):
        test[0,i,0]=1/3
        test[1,i,0]=2/3
        test[2,i,0]=1
        test[:,i,1]=lower+i*((upper-lower)/200)
    
    test=test.to(device)
    phis=net(test)
    
    plt.figure()
    plt.plot(x, phis[0,:,0].cpu().detach().numpy())
    plt.plot(x,x) 
    plt.savefig("plots/s2nr_"+str(s2n_ratio)+"/filter_kappa_1.png",bbox_inches='tight', pad_inches=0.1,  dpi=300)
    plt.show()
    
    plt.figure()
    plt.plot(x, phis[1,:,0].cpu().detach().numpy())
    plt.plot(x,x)    
    plt.savefig("plots/s2nr_"+str(s2n_ratio)+"/filter_kappa_2.png",bbox_inches='tight', pad_inches=0.1,  dpi=300)
    plt.show()
    
    plt.figure()
    plt.plot(x, phis[2,:,0].cpu().detach().numpy())
    plt.plot(x,x)    
    plt.savefig("plots/s2nr_"+str(s2n_ratio)+"/filter_kappa_3.png",bbox_inches='tight', pad_inches=0.1,  dpi=300)
    plt.show()
    
    torch.save(net, 'nets_denoising/s2nr_'+str(s2n_ratio)+'model.pth')
    


