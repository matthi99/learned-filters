# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:24:20 2022

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



#files
filelist=[]
folder = "MRT_Bilder/"
patients = [d for d in os.listdir(folder) if os.path.isdir(folder + d)]
for patient in patients:
    datadir = os.path.join(folder, patient)
    files = os.listdir(datadir)
    for file in files:
        filelist.append(os.path.join(datadir, file))
                

        
data={}
sigma=2
wave= 'sym10'
    
    
#images
width=400
height=400
angles=720
fs=[]
    
for file in filelist:
    x = pydicom.dcmread(file).pixel_array
    x = (x- np.mean(x))/np.std(x)
    #x=x/np.max(x)
    temp=x.shape
    x=np.pad(x,((int((width-temp[0])/2),int((height-temp[0])/2)), (int((width-temp[1])/2),int((height-temp[1])/2))))
    fs.append(x)
    
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
    
np.save('Data_sigma_'+str(sigma)+'.npy', data)