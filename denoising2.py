# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:48:28 2022

@author: Schwab Matthias
"""


import numpy as np
import torch
import pywt
import ptwt


from matplotlib import pyplot as plt

import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader, TensorDataset

import os

import cv2
#from mpl_toolkits.axes_grid1 import ImageGrid



s2n_ratios=np.array([1,2,4,8])
N_epochs=20
wave= 'haar'
n_train=900

folder = "data/faces/"
filelist = [d for d in os.listdir(folder)]
#filelist=filelist[0:n_train]

s2n_ratio=2

level0=[]
level1=[]
level2=[]
gesamt=[]


mean_ideal_risk=0        

for file in filelist:
    x=cv2.imread(folder+file,cv2.IMREAD_GRAYSCALE)
    x=x/255
    x=x-(1/2)
    x=2*x
    image=cv2.resize(x, (256,256))
    sigma=0.1
    noisy_image=image+sigma*np.random.standard_normal(image.shape)
    # print("Difference between image and noisy image:", np.sum((image-noisy_image)**2))
    # wavelet=pywt.wavedec2(image, wave, level=2)
    # DWT_matrix, coeff_slices = pywt.coeffs_to_array(wavelet)
    
    # ideal_risk=np.zeros(DWT_matrix.shape)
    # for i in range(ideal_risk.shape[0]):
    #     for j in range(ideal_risk.shape[1]):
    #         ideal_risk[i,j]=min(DWT_matrix[i,j]**2,sigma**2)
    
    # ideal_risk=np.sum(ideal_risk)
    # print("Ideal risk:", ideal_risk/(256**2))
    # mean_ideal_risk+=ideal_risk/(256**2)
    
    # noisy_wavelet=pywt.wavedec2(noisy_image, wave, level=2)
    # noisy_DWT_matrix, coeff_slices = pywt.coeffs_to_array(noisy_wavelet)
    
    # #print("Difference between DWT and noisy DWT:", np.sum((DWT_matrix-noisy_DWT_matrix)**2))
    
    # #calculate oracle reconstruction
    # noisy_DWT_matrix[abs(DWT_matrix)<sigma]=0
    # rec_coeffs = pywt.array_to_coeffs(noisy_DWT_matrix, coeff_slices, output_format='wavedec2')
    # oracle_rec=pywt.waverec2(rec_coeffs, wave)
    
    # print("Difference between image and oracle reconstruction:", np.sum((image-oracle_rec)**2))
    
    theta=pywt.wavedec2(image, wave, level=2)
    theta_matrix,_ = pywt.coeffs_to_array(theta)
    gesamt.append(theta_matrix)
    level0.append(theta[0])
    level1.append(theta[1][0])
    level1.append(theta[1][1])
    level1.append(theta[1][2])
    level2.append(theta[2][0])
    level2.append(theta[2][1])
    level2.append(theta[2][2])

print(mean_ideal_risk/len(filelist))

def get_max(level):
    maxis=[]
    minis=[]    
    for i in range(len(level)):
        maxis.append(np.max(level[i]))    
        minis.append(np.min(level[i]))    
    upper=np.max(maxis)
    lower=np.min(minis)
    return lower, upper

def create_hist(level, ranges):
    hist=np.zeros(256)
    N=level[0].shape[0]*level[0].shape[1]
    for i in range (len(level)):
        temp=np.histogram(level[i], bins=256, range=ranges)
        hist+=temp[0]
    hist=hist/(N*len(level))
    return temp[1][0:-1], hist



plt.figure(1)
x,y=create_hist(level0,get_max(level0))    
plt.plot(x,y)

plt.figure(2)
x,y=create_hist(level1,get_max(level1))  
plt.plot(x,y)

plt.figure(3)
x,y=create_hist(level2,get_max(level2))    
plt.plot(x,y)

plt.figure(4)
x,y=create_hist(gesamt,get_max(gesamt))    
plt.plot(x,y)
    




    
# images=torch.from_numpy(images.astype(np.float32))

# dataset=TensorDataset(images,images)
# dataloader=DataLoader(dataset,batch_size=1,shuffle=True)


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
