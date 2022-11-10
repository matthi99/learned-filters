# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:19:16 2022

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

from functions import preprocess




                

        
data={}
sigma=8
wave= 'sym10'
    
    
#images
width=400
height=400
angles=720
fs=[]
N=100
    
for i in range(N):
    image=shepp_logan_phantom()
    image=(image-1/2)*2
    for k in range(400):
        for l in range(400):
            if (200-k)**2+(200-l)**2>=120**2 and image[k,l]==-1:
                image[k,l]=0
    fs.append(image)
    
N=len(fs)
fs=np.reshape(fs, (len(fs), width,height))    
data['images']=fs
    
    
#coefficients
coeffs=[]
    
for j in range(N):
    f=fs[j]
    g=radon(f,np.arange(angles)/(angles/180))
    noise=sigma*np.random.randn(width,angles)
    gdelta = g + noise
    f_FBP=iradon(gdelta)
    coeff, L1, L2, =preprocess(f_FBP, wave)
    coeffs.append(coeff)
    if j%10==0:
        print(j)
        
coeffs=np.reshape(coeffs,(len(coeffs), coeffs[0].shape[0],coeffs[0].shape[1]))
    
#save
data['coefficients']=coeffs
    
data['upper']=np.max(coeffs)
data['lower']=np.min(coeffs)
    
data['L1']=L1
data['L2']=L2
    
np.save('Data_shepp_sigma_'+str(sigma)+'.npy', data)