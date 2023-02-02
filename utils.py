# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:58:48 2023

@author: matthias
"""

import numpy as np
import torch
import os
import torch.nn as nn
import ptwt
from matplotlib import pyplot as plt


class Dataset(torch.utils.data.Dataset):
    """
    Dataset for the depth/text data.
    This dataset loads an image and applies some transformations to it.
    """

    def __init__(self, folder= "C:/Users/matthias/Desktop/Data/CT-for-filters/preprocessed/", s2n_ratio=128, 
                 train=True, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train=train
        if self.train:
            self.folder = folder+"train/"
        else:
            self.folder = folder+"test/"
        self.examples = self.get_examples()
        self.s2n_ratio=s2n_ratio
        
        

    def __getitem__(self,idx):
        data = np.load(os.path.join(self.folder, self.examples[idx]), allow_pickle=True).item()
        
        s2n_index=np.where(data['signal_to_noise']==self.s2n_ratio)[0][0]
        inp=data['x_fbp'][s2n_index]
        inp=np.expand_dims(inp,0)
        outp=data['x']
        outp=np.expand_dims(outp,0)
                            
        inp = torch.from_numpy(inp.astype("float32")).to(self.device)
        outp = torch.from_numpy(outp.astype("float32")).to(self.device)
        return inp, outp

    def __len__(self):
        return len(self.examples)
    

    def get_examples(self):
        examples = [f for f in os.listdir(self.folder) if f.endswith('.npy')]
        return examples

class Collator:
    """
    Data collator for different types of mask-combinations.
    """


    def __call__(self, batch, *args, **kwargs):
        inputs = []
        outputs = []
        
        for inp, outp in batch:
            inputs.append(inp)
            outputs.append(outp)
        return torch.stack(inputs), torch.stack(outputs)


class learned_filter(torch.nn.Module):

    def __init__(self, indim, outdim):
        super().__init__()  
        self.lin1 = nn.Linear(in_features=indim, out_features=20)
        self.lin2 = nn.Linear(in_features=20, out_features=40)
        self.lin3 = nn.Linear(in_features=40, out_features=20)
        self.lin4 = nn.Linear(in_features=20, out_features=20)
        self.lin5 = nn.Linear(in_features=20, out_features=outdim)
        self.flat = nn.Flatten()
        self.act1 = nn.Tanh()
        self.actr = nn.ReLU()


    def forward(self, inp):
        xx = self.actr(self.lin1(inp))
        xx = self.actr(self.lin2(xx))
        xx = self.actr(self.lin3(xx))
        xx = self.act1(self.lin4(xx))
        out = self.lin5(xx)
        return  out

def get_histogram():
    hist={}
    hist['trainloss']=[]
    hist['testloss']=[]
    hist['fbploss']=[]
    return hist

def reconstruct(nets, x_fbp, levels, wave):
    coefficients= ptwt.wavedec2(x_fbp, wave, level=levels)
    denoised_coeff=[]
    denoised_coeff.append(coefficients[0])
    for i in range(levels):
        inp=torch.flatten(torch.stack([c for c in coefficients[i+1]]))
        filtered=nets[i](inp[:,None])
        filtered_reshaped=[]
        shape=coefficients[i+1][0].shape
        lenght=shape[-1]*shape[-2]
        for k in range(3):
            filtered_reshaped.append(torch.reshape(filtered[k*lenght:(k+1)*lenght,0], shape))
        denoised_coeff.append(filtered_reshaped)
        rec=ptwt.waverec2(denoised_coeff, wave)
    return rec

def save_checkpoint(nets,s2n_ratio, wave, best):
    if not os.path.exists('RESULTS_FOLDER/'+wave+'/'):
        os.makedirs('RESULTS_FOLDER/'+wave+'/')
    if not os.path.exists('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/'):
        os.makedirs('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/')
    if not os.path.exists('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/weights/'):
        os.makedirs('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/weights/')
    for i in range(len(nets)):
        savepath= 'RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/weights/level_'+str(i+1)+'_'+best+'.pth'
        torch.save(nets[i].state_dict(), savepath)


def plot_filter(nets, s2n_ratio, wave, device):
    if not os.path.exists('RESULTS_FOLDER/'+wave+'/'):
        os.makedirs('RESULTS_FOLDER/'+wave+'/')
    if not os.path.exists('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/'):
        os.makedirs('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/')
    if not os.path.exists('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/plots/'):
        os.makedirs('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/plots/')
    for i in range(len(nets)):
            savepath='RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/plots/level_'+str(i+1)+'.png'
            j=i+1 #der Scale index: -j 
            kappa=2**(-j/2) #quasi-singulär-wert 
            x=kappa*torch.linspace(-10, 10, 100, device=device)
            x=x[:,None]
            y=kappa*nets[i](x/kappa)
            plt.figure()
            plt.plot(x[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy())
            plt.plot(x[:,0].cpu().detach().numpy(),x[:,0].cpu().detach().numpy())
            plt.title('Level:'+str(i+1))
            plt.savefig(savepath, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()

def plot_hist(hist,s2n_ratio, wave):
    if not os.path.exists('RESULTS_FOLDER/'+wave+'/'):
        os.makedirs('RESULTS_FOLDER/'+wave+'/')
    if not os.path.exists('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/'):
        os.makedirs('RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/')
    savepath='RESULTS_FOLDER/'+wave+'/'+'s2nr_'+str(s2n_ratio)+'/train_progress.png'
    plt.figure()
    plt.plot(hist['trainloss'])
    plt.plot(hist['testloss'])
    plt.plot(hist['fbploss'])
    plt.savefig(savepath, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    train_data = Dataset(s2n_ratio=8)    
    collator = Collator()
    dataloader= torch.utils.data.DataLoader(train_data, batch_size=1, collate_fn=collator,shuffle=True) 
    
    i=0
    for x_fbp, x in dataloader:
        i+=1
        if i%100==0:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(x[0,0,...].cpu())
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(x_fbp[0,0,...].cpu())
            plt.axis('off')
            plt.show()
            

    