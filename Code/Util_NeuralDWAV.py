import numpy as np
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator_test():#Stepwise function with gaussian noise
    def __init__(self, MySig):
        np.random.seed(42)
        self.Nchange=5
        self.Np=2**13
        self.sigma=0.2
        
    def __getitem__(self, batchIndex):
        batchX, batchY = self.generate_batch(batchIndex)
        return torch.tensor(batchX[:,np.newaxis,:]).to(device).double(), torch.tensor(batchY[:,np.newaxis,:]).to(device).double()

    def generate_batch(self, batchIndex):
        batchX=np.zeros((batchIndex,self.Np))
        batchY=np.zeros((batchIndex,self.Np))
        for batch in range(batchIndex):
            x=np.random.randn(self.Nchange)
            y=np.concatenate(([0],np.floor(np.sort(np.random.rand(self.Nchange))*self.Np).astype(int)))
            for i in range(self.Nchange):
                batchY[batch][y[i]:y[i+1]]=x[i]
            batchX[batch]=batchY[batch]+ self.sigma*np.random.randn(self.Np)
        return batchX, batchY


def npt(x):
    return x.detach().numpy().squeeze()

def plt_x(n,X,L):
    plt.figure(n)
    for i in range(len(X)):
        plt.plot(npt(X[i]))
    plt.legend(L)
    
def plt_Emb(n,X,L):
    plt.figure(n,figsize=(8,12))
    fig, axs = plt.subplots(2)
    for i in range(len(X)):
        print(len(X))
        axs[i].imshow(20*np.log10(npt(X[i])+1e-3),aspect="auto")
        axs[i].set_title(L[i])
