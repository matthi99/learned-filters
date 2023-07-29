# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:01:49 2023

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


parser = argparse.ArgumentParser(description= 'Define parameters for testing. Of course only parameters were trained on are allowed!')



parser.add_argument('--wave', help="Define which wavelet transform should be used", type=str, default="haar")
parser.add_argument('--levels', help="Number of levels in wavelet transform. Has to be integer between 1 and 8", 
                    type=int, default=8)
parser.add_argument('--s2n_ratio', help="possible signal-to-noise ratios are: 2,4,8,16,32,64,128,256,512", 
                    type=int, default=4)
parser.add_argument('--noise', help= "Noise type", type=str, default ="uniform")

args = parser.parse_args()



wave= args.wave
levels=args.levels #Maximal 8 possible levels
s2n_ratio=args.s2n_ratio
noise = args.noise


f = open("path_to_data.txt", "r")
folder=f.read()
folder=folder+ '/non-COVID/'

savefolder="RESULTS_FOLDER/"+ wave +"/"+ noise +"/" +'s2nr_'+str(s2n_ratio) +"/results/"
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
angles=512
filelist=os.listdir(folder) 


device = "cuda" if torch.cuda.is_available() else "cpu"

best='best'
nets=[]
for i in range(levels):
    loadpath = 'RESULTS_FOLDER/'+wave+'/'+noise+'/'+'s2nr_'+str(s2n_ratio)+'/weights/level_'+str(i+1)+'_'+best+'.pth'
    weights = torch.load(loadpath, map_location=torch.device(device))
    net=learned_filter(1,1).to(device)
    net.load_state_dict(weights)
    net.eval()
    nets.append(net)

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
            



results={}
MSE=[]
MSE_fbp=[]
MAE=[]
MAE_fbp=[]

for file in tqdm(filelist):
    x=cv2.imread(folder+file,cv2.IMREAD_GRAYSCALE)
    x=x/255
    x=x-(1/2)
    x=2*x
    x=cv2.resize(x, (256,256))
    y=radon(x,np.arange(angles)/(angles/180), circle=False)
    if noise =="gaussian":
        sigma=np.sqrt(np.mean(y**2)/s2n_ratio)
        z=sigma*np.random.randn(y.shape[0], angles)
    elif noise =="poisson":
        m=np.min(y)
        y=y-m
        z=np.random.poisson(y)-y
        y=y+m
        scale=np.mean(y**2)/(np.mean(z**2)*s2n_ratio)
        z=z*np.sqrt(scale)
    elif noise =="uniform":
        a=np.sqrt((3*np.mean(y**2))/s2n_ratio)
        z= np.random.uniform(low=-a, high=a, size=(y.shape[0], angles))
    elif noise == "saltpepper":
        z=np.zeros_like(y)
        ma=np.max(y)
        mi=np.min(y)
        while np.mean(z**2)<np.mean(y**2)/s2n_ratio:
            x_coord=np.random.randint(0, y.shape[0])
            y_coord=np.random.randint(0, angles)
            if np.random.uniform() <0.5:
                z[x_coord,y_coord]=mi-y[x_coord,y_coord]
            else:
                z[x_coord,y_coord]=ma-y[x_coord,y_coord]
    else: 
        print("Wrong argument for --noise!")
    y_delta=y+z
    x_fbp=iradon(y_delta, circle=False)
    inp=np.expand_dims(x_fbp,0)
    inp = torch.from_numpy(inp.astype("float32")).to(device)
    rec=reconstruct(nets, inp, levels, wave)
    rec=rec[0,0,...].cpu().detach().numpy()
    print(rec.shape,x.shape, x_fbp.shape)
    
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(x, cmap='gray',vmin=-1.5,vmax=1.5)
    axes[0].axis('off')
    axes[0].title.set_text('Ground truth')
    axes[1].imshow(x_fbp, cmap='gray', vmin=-1.5,vmax=1.5)
    axes[1].axis('off')
    axes[1].title.set_text('FBP')
    im=axes[2].imshow(rec, cmap='gray', vmin=-1.5,vmax=1.5)
    axes[2].axis('off')
    axes[2].title.set_text('Method')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.33, 0.05, 0.35])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(savefolder+file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    
    results[file]={}
    results[file]['Mean squared error']=np.mean((x-rec)**2)
    MSE.append(np.mean((x-rec)**2))
    results[file]['Mean squared error (fbp)']=np.mean((x_fbp-rec)**2)
    MSE_fbp.append(np.mean((x_fbp-rec)**2))
    results[file]['Mean absolute error']=np.mean(abs(x-rec))
    MAE.append(np.mean(abs(x-rec)))
    results[file]['Mean absolute error (fbp)']=np.mean(abs(x_fbp-rec))
    MAE_fbp.append(np.mean(abs(x_fbp-rec)))

results['mean']={}
results['mean']['Mean squared error']=np.mean(MSE)
results['mean']['Mean squared error (fbp)']=np.mean(MSE_fbp)
results['mean']['Mean absolute error']=np.mean(MAE)
results['mean']['Mean absolute error (fbp)']=np.mean(MAE_fbp)

with open(savefolder+'summary.txt', 'w') as file:
     json.dump(results, file, indent=4, sort_keys=False)

