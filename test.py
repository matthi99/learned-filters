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


parser.add_argument('input_folder', help="Path to folder were test data is stored", type=str)
parser.add_argument('output_folder', help="Path to folder were results should be stored", type=str)
parser.add_argument('--wave', help="Define which wavelet transform should be used", type=str, default="haar")
parser.add_argument('--levels', help="Number of levels in wavelet transform. Has to be integer between 1 and 8", 
                    type=int, default=8)
parser.add_argument('--s2n_ratio', help="possible signal-to-noise ratios are: 2,4,8,16,32,64,128,256,512", 
                    type=int, default=8)
parser.add_argument('--noise', help= "Noise type", type=str, default ="gaussian")

args = parser.parse_args()

folder = args.input_folder+'/'
savefolder=args.output_folder+'/'

wave= args.wave
levels=args.levels #Maximal 8 possible levels
s2n_ratio=args.s2n_ratio
noise = args.noise

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
    sigma=np.sqrt(np.mean(y**2)/s2n_ratio)
    z=sigma*np.random.randn(y.shape[0], angles)
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

