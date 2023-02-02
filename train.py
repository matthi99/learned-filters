# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:23:29 2023

@author: Schwab Matthias
"""


import numpy as np
import torch


import shutil
import os
import argparse

from utils import *

#define parameters
parser = argparse.ArgumentParser(description= 'Define parameters for training')

parser.add_argument('--wave', help="Define which wavelet transform should be used", type=str, default="haar")
parser.add_argument('--levels', help="Number of levels in wavelet transform. Has to be integer between 1 and 8", 
                    type=int, default=8)
parser.add_argument('--s2n_ratio', help="possible signal-to-noise ratios are: 2,4,8,16,32,64,128,256,512", 
                    type=int, default=8)
parser.add_argument('--N_epochs', help="Specify how many epochs should be trained", type=int, default=100)
args = parser.parse_args()


wave= args.wave
levels=args.levels #Maximal 8 possible levels
s2n_ratio=args.s2n_ratio #possible signal to noise ratios are: 2,4,8,16,32,64,128,256,512 
N_epochs=args.N_epochs

#get path to preprocessed data
f = open("path_to_data.txt", "r")
folder=f.read()

if not os.path.exists('RESULTS_FOLDER/'):
    os.makedirs('RESULTS_FOLDER/')

#delete old results
if os.path.exists('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/'):
    shutil.rmtree('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/')

#get dataloaders
train_data = Dataset(folder=folder, s2n_ratio=s2n_ratio)  
test_data = Dataset(folder=folder, s2n_ratio=s2n_ratio, train=False)    
collator = Collator()
dataloader= torch.utils.data.DataLoader(train_data, batch_size=1, collate_fn=collator,shuffle=True)
dataloader_test= torch.utils.data.DataLoader(test_data, batch_size=1, collate_fn=collator,shuffle=False)     


#Prepare networks for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

hist=get_histogram()
best_loss=np.inf
    
nets=[]
for i in range(levels):
    nets.append(learned_filter(1,1).to(device))

params=[]
for i in range(levels):
    d={}
    d['params']=nets[i].parameters()
    d['lr']=1e-3
    params.append(d)

lambda1 = lambda epoch: (1-epoch/N_epochs)**0.9    
optimizer = torch.optim.AdamW(params, weight_decay=1e-3,)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1 for i in range(levels)])

#train
for epoch in range(N_epochs):
        print(f"Epoch {epoch}/{N_epochs}\n-------------------------------")
        for net in nets:
            net.train()
        count=0
        l=0
        for x_fbp, x in dataloader:
            optimizer.zero_grad()
            rec=reconstruct(nets, x_fbp, levels, wave)
            loss=torch.mean((rec-x)**2)
            loss.backward()
            for i in range(levels):
                torch.nn.utils.clip_grad_norm_(params[i]['params'], 2)
            optimizer.step()
            l+=loss.item()
            count+=1
        hist['trainloss'].append(l/count)
        scheduler.step()
        if l < best_loss:
            print(f"New best loss: {l/count}, -->Saving models")
            best_loss=l
            save_checkpoint(nets,s2n_ratio, wave, 'best')
            plot_filter(nets, s2n_ratio, wave, device)
        #print(scheduler.get_last_lr())
        # for i in range(levels):
        #     print(torch.sum(nets[i].lin1.weight.grad**2))
        
        #validation
        for net in nets:
            net.eval()
        count=0
        l=0
        n=0
        for x_fbp, x in dataloader_test:
            with torch.no_grad():
                rec=reconstruct(nets, x_fbp, levels, wave)
                loss=torch.mean((rec-x)**2)
                l+=loss.item()
                n+=torch.mean((x_fbp-x)**2).item()
                count+=1
            
        hist['testloss'].append(l/count)
        hist['fbploss'].append(n/count)
        
        #print('Mean rec error on train data set:', hist['trainloss'][-1])
        #print('Mean rec error on test data set:', hist['testloss'][-1])
        plot_hist(hist,s2n_ratio, wave)

#save histogram and weights
np.save('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/histogram.npy', hist)
save_checkpoint(nets,s2n_ratio, wave, 'last')    

      
        



