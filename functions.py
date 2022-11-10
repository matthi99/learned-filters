# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 22:27:03 2022

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
import random

def preprocess(bad_rec, wave):
    a,[av,ah,ad], [v,h,d]=(pywt.wavedec2(bad_rec, wave, level=2)) 
    
    L1=np.prod(a.shape)
    a=np.reshape(a, (L1))
    
    av=np.reshape(av,(L1))
    ah=np.reshape(ah,(L1))
    ad=np.reshape(ad,(L1))
    
    L2=np.prod(v.shape)
    v=np.reshape(v,(L2))
    h=np.reshape(h,(L2))
    d=np.reshape(d,(L2))
    L=4*L1+3*L2
    
    temp=np.zeros((L,2))
    for i in range(L1):
        temp[i,0]=1/3
        temp[i,1]=a[i]
    for i in range(L1):
        temp[L1+i,0]=2/3
        temp[L1+i,1]=av[i]
    for i in range(L1):
        temp[2*L1+i,0]=2/3
        temp[2*L1+i,1]=ah[i]
    for i in range(L1):
        temp[3*L1+i,0]=2/3
        temp[3*L1+i,1]=ad[i]
    for i in range(L2):
        temp[4*L1+i,0]=1
        temp[4*L1+i,1]=v[i]
    for i in range(L2):
        temp[4*L1+L2+i,0]=1
        temp[4*L1+L2+i,1]=h[i]
    for i in range(L2):
        temp[4*L1+2*L2+i,0]=1
        temp[4*L1+2*L2+i,1]=d[i]
    return temp, L1, L2


def reshape(coeff_reg, L1, L2):
    s1=int(np.sqrt(L1))
    s2=int(np.sqrt(L2))
    a=coeff_reg[0,0:L1,0]
    a=torch.reshape(a,(s1,s1))
    a=a[None, None, :, :]
    
    av=coeff_reg[0,L1:2*L1,0]
    av=torch.reshape(av,(s1,s1))
    av=av[None, None, :, :]
    
    ah=coeff_reg[0,2*L1:3*L1,0]
    ah=torch.reshape(ah,(s1,s1))
    ah=ah[None, None, :, :]
        
    ad=coeff_reg[0,3*L1:4*L1,0]
    ad=torch.reshape(ad,(s1,s1))
    ad=ad[None, None, :, :]
        
    v=coeff_reg[0,4*L1:4*L1+L2,0]
    v=torch.reshape(v,(s2,s2))
    v=v[None, None, :, :]
        
    h=coeff_reg[0,4*L1+L2:4*L1+2*L2,0]
    h=torch.reshape(h,(s2,s2))
    h=h[None, None, :, :]
        
    d=coeff_reg[0,4*L1+2*L2:4*L1+3*L2,0]
    d=torch.reshape(d,(s2,s2))
    d=d[None, None, :, :]
    return  a, [av, ah, ad], [v,h,d]

def create_random_phantom(circle,ellipse,square):
    n_circle=random.randint(0,circle)
    n_ellipse=random.randint(0,ellipse)
    n_square=random.randint(0,square)
    image=np.zeros((400,400))
    for i in range(n_circle):
        r=np.random.randint(0,100)
        center=np.array([np.random.randint(150,250), np.random.randint(150,250)])
        intensity=np.random.uniform(0, 1)
        for k in range(400):
            for j in range(400):
                if (center[0]-k)**2+(center[1]-j)**2<=r**2:
                    image[k,j]+=intensity
    for i in range(n_ellipse):
        a=np.random.randint(0,100)
        b=np.random.randint(0,100)
        center=np.array([np.random.randint(150,250), np.random.randint(150,250)])
        intensity=np.random.uniform(0, 1)
        for k in range(400):
            for j in range(400):
                if ((center[0]-k)/a)**2+((center[1]-j)/b)**2<=1:
                    image[k,j]+=intensity
    return image


    
                
        
        
    
    
    
    
    
    
    