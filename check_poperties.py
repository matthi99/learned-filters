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
alphas=[32,16,8,4,2]

folder=os.path.join("RESULTS_FOLDER", wave, noise)

#Definition 2.1 Properties 1, 2, 3
#check for all kappas (levels) 

for level in range(1,9,1):
    kappa=2**(-level/2)
    x=kappa*torch.linspace(-2, 2, 1000, device=device)
    x=x[:,None]
    plt.figure()
    plt.plot(x[:,0].cpu().detach().numpy(),x[:,0].cpu().detach().numpy(), label="Id")
    for alpha in alphas:
        loadpath=os.path.join(folder, f"alpha_{alpha}", "weights", f"level_{level}_best.pth")
        weights = torch.load(loadpath, map_location=torch.device(device))
        net=learned_filter(1,1).to(device)
        net.load_state_dict(weights)
        net.eval()
        y=kappa*net(x/kappa)
        plt.plot(x[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy(), label=f"$\\alpha$={alpha}")
        
    plt.title(f"Filters for {noise} noise and $\kappa$ = {round(kappa,4)}")
    plt.legend(loc ="lower right")
    plt.savefig(os.path.join(folder,f"Filters_kappa_{level}.png") , dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()



#Property 1 ---> strictly increasing

for level in range(1,9,1):
    kappa=2**(-level/2)
    x=kappa*torch.linspace(-2, 2, 100, device=device)
    x=x[:,None]
    for alpha in alphas:
        loadpath=os.path.join(folder, f"alpha_{alpha}", "weights", f"level_{level}_best.pth")
        weights = torch.load(loadpath, map_location=torch.device(device))
        net=learned_filter(1,1).to(device)
        net.load_state_dict(weights)
        net.eval()
        y=kappa*net(x/kappa)
        gradient= np.gradient(y[:,0].cpu().detach().numpy())
        if np.min(gradient)<0:
            print(f"Not strictly increasing for {noise} noise, kappa = {kappa} and alpha = {alpha}.", 
                  f"Amount of decreasing is {np.min(gradient)}")


#Property 2 --> continuous ->clear!

#Property 3 --> |phi(x)| < |x|

for level in range(1,9,1):
    kappa=2**(-level/2)
    x=kappa*torch.linspace(-2, 2, 100, device=device)
    x=x[:,None]
    for alpha in alphas:
        loadpath=os.path.join(folder, f"alpha_{alpha}", "weights", f"level_{level}_best.pth")
        weights = torch.load(loadpath, map_location=torch.device(device))
        net=learned_filter(1,1).to(device)
        net.load_state_dict(weights)
        net.eval()
        y=kappa*net(x/kappa)
        abs_y= abs(y[:,0].cpu().detach().numpy())
        abs_x= abs(x[:,0].cpu().detach().numpy())
        if np.max(abs_y-abs_x)>0:
            print(f"Not smaller then identity for {noise} noise, kappa = {kappa} and alpha = {alpha}.", 
                  f"Amount of offset is {np.max(abs_y-abs_x)}")

#Property 4 --> phi(x) --> x for alpha -->0
plt.figure()
for level in range(1,9,1):
    kappa=2**(-level/2)
    x=kappa*torch.linspace(-2, 2, 100, device=device)
    x=x[:,None]
    diff=[]
    for alpha in alphas:
        loadpath=os.path.join(folder, f"alpha_{alpha}", "weights", f"level_{level}_best.pth")
        weights = torch.load(loadpath, map_location=torch.device(device))
        net=learned_filter(1,1).to(device)
        net.load_state_dict(weights)
        net.eval()
        y=kappa*net(x/kappa)
        y_np=y[:,0].cpu().detach().numpy()
        x_np= x[:,0].cpu().detach().numpy()
        delta_x=x_np[1]-x_np[0]
        diff.append(np.sqrt(np.sum((y_np-x_np)**2)*delta_x))
    plt.plot(alphas, diff, label=f"$\kappa$={round(kappa,2)}")
        
plt.legend(loc ="upper right")
#plt.savefig(os.path.join(folder,f"Filters_kappa_{level}.png") , dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
    
#Assumption A. 1.



#Assumption A. 2.

plt.figure()
for level in range(1,9,1):
    kappa=2**(-level/2) 
    max_ratios=[]
    for alpha in alphas:
        loadpath=os.path.join(folder, f"alpha_{alpha}", "weights", f"level_{level}_best.pth")
        weights = torch.load(loadpath, map_location=torch.device(device))
        net=learned_filter(1,1).to(device)
        net.load_state_dict(weights)
        net.eval()
        c=1/100
        d=300
        x=torch.linspace(-c*alpha/kappa, c*alpha/kappa, 100, device=device)
        x=x[:,None]
        y=kappa*net(x/kappa)
        y_abs=abs(y[:,0].cpu().detach().numpy())
        x_abs=abs(x[:,0].cpu().detach().numpy())
        ratio = (y_abs/x_abs)*(alpha/(d*kappa))
        max_ratios.append(np.max(ratio))
    plt.plot(alphas,max_ratios, label=f"$\kappa$={round(kappa,2)}")

plt.legend(loc ="upper right")
#plt.savefig(os.path.join(folder,f"Filters_kappa_{level}.png") , dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
    
        # plt.plot(x[:,0].cpu().detach().numpy(),(y[:,0].cpu().detach().numpy()/x[:,0].cpu().detach().numpy())*alpha/kappa)
        # #plt.plot(x[:,0].cpu().detach().numpy(),kappa/alpha*np.ones(1000))
        # # plt.plot(x[:,0].cpu().detach().numpy(),(kappa/alpha)*x[:,0].cpu().detach().numpy())
        # # plt.plot(x[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy(), label=r"$\alpha$"+f"= {alpha}")
        # plt.title(f"$\kappa$ = {kappa}")
        # plt.legend(loc ="lower right")
        # plt.show()
        
        
