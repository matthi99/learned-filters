# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:44:00 2023

@author: matthias
"""

import numpy as np
import torch


from skimage.transform import radon, iradon
from matplotlib import pyplot as plt


import os
import argparse
from tqdm import tqdm
import cv2
from utils import *
import json



#Eigenschaften überprüfen
for i in range(len(nets)):
    j=i+1 #der Scale index: -j 
    kappa=2**(-j/2) #quasi-singulär-wert 
    x=kappa*torch.linspace(-20, 20, 100, device=device)
    x=x[:,None]
    y=kappa*nets[i](x/kappa)
    x=x[:,0].cpu().detach().numpy()
    y=y[:,0].cpu().detach().numpy()
    h=x[1]-x[0]
    dy=np.zeros(100)
    for i in range(1,99,1):
        dy[i]=(y[i+1]-y[i-1])/(2*h)
    print(np.max(dy))
            
