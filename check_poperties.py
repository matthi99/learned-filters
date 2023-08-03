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


wave="haar"
noise="gaussian"
device = "cuda" if torch.cuda.is_available() else "cpu"

folder=os.path.join("RESULTS_FOLDER", wave, noise)

#Definition 3.2 Properties 1 and 2
#check for smallest all kappas (levels) 

for level in range(1,9,1):
    kappa=2**(-level/2)
    x=kappa*torch.linspace(-2, 2, 1000, device=device)
    x=x[:,None]
    plt.figure()
    plt.plot(x[:,0].cpu().detach().numpy(),x[:,0].cpu().detach().numpy(), label="Id")
    for i in range(5,10):
        loadpath=os.path.join(folder, f"s2nr_{2**i}", "weights", f"level_{level}_best.pth")
        weights = torch.load(loadpath, map_location=torch.device(device))
        net=learned_filter(1,1).to(device)
        net.load_state_dict(weights)
        net.eval()
        y=kappa*net(x/kappa)
        plt.plot(x[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy(), label=f"SNR={2**i}")
        
    plt.title(f"Filters for {noise} noise and $\kappa$ = {round(kappa,4)}")
    plt.legend(loc ="lower right")
    plt.savefig(os.path.join(folder,f"Filters_kappa_{level}.png") , dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()


#Definition 3.2 Properties 3
#check for kappa smaller than 1/4 --> gamma=16

l_alpha=0.999
d=1/5
for level in range(4,9,1):
    kappa=2**(-level/2) 
    constant=(gamma*kappa**2*l_alpha)/(1-l_alpha*(1-gamma*kappa**2))
    print(constant)
   
    for i in range(4,10):
        loadpath=os.path.join(folder, f"s2nr_{2**i}", "weights", f"level_{level}_best.pth")
        weights = torch.load(loadpath, map_location=torch.device(device))
        net=learned_filter(1,1).to(device)
        net.load_state_dict(weights)
        net.eval()
        alpha=1/(2**i)
        x=torch.linspace(-d*alpha/kappa, d*alpha/kappa, 1000, device=device)
        x=x[:,None]
        y=kappa*net(x/kappa)
        plt.figure()
        plt.plot(x[:,0].cpu().detach().numpy(),constant*x[:,0].cpu().detach().numpy())
        plt.plot(x[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy(), label=r"$\alpha$"+f"= {alpha}")
        plt.title(f"$\kappa$ = {kappa}")
        plt.legend(loc ="lower right")
        plt.show()
        
        
#check L-continous filters
#check for kappa all kappa
L_constants={}
for level in range(4,9,1):
    kappa=2**(-level/2) 
    L_constants[f"Kappa={kappa}"]=[]
   
    for i in range(1,10):
        loadpath=os.path.join(folder, f"s2nr_{2**i}", "weights", f"level_{level}_best.pth")
        weights = torch.load(loadpath, map_location=torch.device(device))
        net=learned_filter(1,1).to(device)
        net.load_state_dict(weights)
        net.eval()
        
        alpha=1/2**i
        x=kappa*torch.linspace(-2, 2, 100, device=device)
        x=x[:,None]
        y=kappa*net(x/kappa)
        
        x=x[:,0].cpu().detach().numpy()
        y=y[:,0].cpu().detach().numpy()
        h=x[1]-x[0]
        dy=np.zeros(len(y))
        for i in range(1,len(y)-1,1):
            dy[i]=(y[i+1]-y[i-1])/(2*h)
        if np.min(dy)<0:
            print(f"Stricly increasing is violated for kappa={round(kappa,4)} and alpha= {round(alpha,4)}!")
        if np.max(abs(y)-abs(x)-0.025)>0:
            print(f"Condition 3 is violated for kappa={round(kappa,4)} and alpha= {round(alpha,4)}")
        L_constants[f"Kappa={kappa}"].append(np.max(dy))
        
for key in L_constants.keys():
    maximum = np.max(L_constants[key])
    argmax=np.argmax(L_constants[key])
    plt.figure()
    plt.plot(L_constants[key])
    plt.show
    
          
 

