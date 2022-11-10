# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:27:49 2022

@author: matth
"""
import numpy as np
import torch
import pywt
import ptwt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon
from matplotlib import pyplot as plt

import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader, TensorDataset

import os
import pydicom

import cv2
from mpl_toolkits.axes_grid1 import ImageGrid

from functions import preprocess
from functions import reshape

sigma=2
N_epochs=50
wave= 'sym10'
data=np.load('Data_sigma_'+str(sigma)+'.npy', allow_pickle=True).item()


coeffs=data['coefficients']
images=data['images']
upper=data['upper']
lower=data['lower']
# L1=data['L1']
# L2=data['L2']
L1=114**2
L2=209**2

coeffs=torch.from_numpy(coeffs.astype(np.float32))
images=torch.from_numpy(images.astype(np.float32))

dataset=TensorDataset(coeffs,images)
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


size = len(dataloader.dataset)
for epoch in range(N_epochs):
    print(f"Epoch {epoch}\n-------------------------------")
    count=0
    l=0
    for coeff, image in dataloader:
        net_optimizer.zero_grad()
        
        coeff_reg=net(coeff)
        a, [av, ah, ad], [v,h,d]=reshape(coeff_reg, L1, L2)
        rec=ptwt.waverec2([a, [av, ah, ad], [v,h,d]], wave)
        
        loss=torch.mean((rec-image)**2)
        loss.backward()
        net_optimizer.step()
        
        l+=loss.item()
        count+=1
    print(f"{'Loss'}={l/count}")



#save network
torch.save(net, 'net_sigma_'+str(sigma))



#plot learned filters for the different kappas
lower=-10
upper=10
x=np.zeros(200)
for i in range(200):
    x[i]=lower+i*((upper-lower)/200)

    

test=torch.zeros((3,200,2))
for i in range(200):
    test[0,i,0]=1/3
    test[1,i,0]=2/3
    test[2,i,0]=1
    test[:,i,1]=lower+i*((upper-lower)/200)
    
phis=net(test)


# j =-((7-2+1)-1+1)
# kappa=2**(j/2)

# y1=kappa*x
# phiy1=kappa*phis[0,:,0].cpu().detach().numpy()

# plt.figure()
# plt.plot(y,phiy)
# plt.plot(y,y)
plt.figure()
plt.plot(x, phis[0,:,0].cpu().detach().numpy())
plt.plot(x,x)    
plt.savefig("sigma_"+str(sigma)+"_kappa_1.png",bbox_inches='tight', pad_inches=0.1,  dpi=300)

# j =-((7-2+1)+0+1)
# kappa=2**(j/2)

# y2=kappa*x
# phiy2=kappa*phis[1,:,0].cpu().detach().numpy()

# plt.figure()
# plt.plot(y,phiy)
# plt.plot(y,y)

plt.figure()
plt.plot(x, phis[1,:,0].cpu().detach().numpy())
plt.plot(x,x)    
plt.savefig("sigma_"+str(sigma)+"_kappa_2.png",bbox_inches='tight', pad_inches=0.1,  dpi=300)
# j =-((7-2+1)+1+1)
# kappa=2**(j/2)

# y3=kappa*x
# phiy3=kappa*phis[2,:,0].cpu().detach().numpy()

# plt.figure()
# plt.plot(y1,phiy1)
# plt.plot(y2,phiy2)
# plt.plot(y2,phiy2)
# plt.plot(y,y)

plt.figure()
plt.plot(x, phis[2,:,0].cpu().detach().numpy())
plt.plot(x,x)    
plt.savefig("sigma_"+str(sigma)+"_kappa_3.png",bbox_inches='tight', pad_inches=0.1,  dpi=300)




#test method on shepp logan phantom
f = shepp_logan_phantom()
g=radon(f,np.arange(720)/4)

wave= 'sym10'
noise=sigma *np.random.randn(400,720)
gdelta=g+noise

f_FBP=iradon(gdelta)

coef, _, _=preprocess(f_FBP, wave) 
coef=np.expand_dims(coef,0)
coef=torch.from_numpy(coef.astype(np.float32))
coef_reg=net(coef)
a, [av, ah, ad], [v,h,d]=reshape(coef_reg, L1, L2)
rec=ptwt.waverec2([a, [av, ah, ad], [v,h,d]], wave)
rec=rec[0,0,:,:].cpu().detach().numpy()


fig = plt.figure(figsize=(9.75, 3))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
maxi=np.max(np.concatenate((f,f_FBP,rec)))
mini=np.min(np.concatenate((f,f_FBP,rec)))

# Add data to image grid
im=grid[0].imshow(f,cmap='jet', vmin=mini, vmax=maxi)
grid[0].axis('off')
im=grid[1].imshow(f_FBP, cmap='jet',vmin=mini, vmax=maxi)
grid[1].axis('off')
im=grid[2].imshow(rec, cmap='jet',vmin=mini, vmax=maxi)
grid[2].axis('off')
# Colorbar
grid[2].cax.colorbar(im)
grid[2].cax.toggle_label(True)

#plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible
plt.savefig("reconstructions_"+str(sigma)+".png",bbox_inches='tight', pad_inches=0.1,  dpi=300)
plt.show()


print(f"{'Error_fbp'}={np.mean((f-f_FBP)**2)}")
print(f"{'Error_learned_filter'}={np.mean((f-rec)**2)}")




