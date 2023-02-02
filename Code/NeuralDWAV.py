import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pywt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuralDWAV(nn.Module):
    def __init__(self, 
                 Input_Size,        ###Mandatory parameter
                 Input_Level=5,     ###Level L, provide L+1 output for LDWT/Despawn and 2**L for the LWPT
                 Input_Archi="WPT", ###Architecture based on Wavelet Packet Transform "WPT"or Discrete Wavelet Transform "DWT".
                 Filt_Trans = True, ### If you want the inverse transform "True" 
                 Filt_Train = True, #or "False" trainable or not
                 Filt_Tfree = False, #or free transposed layer or not
                 Filt_Style = "Filter_Free", #or "Module_Free", "Layer_Free", "Kernel_Free"
                 Filt_Mother = "db4",
                 Act_Train = True, #or "False"
                 Act_Style = "Sigmoid", #or "Soft", "Hard"
                 Act_Symmetric = True, #or False
                 Act_Init = 0):
        super(NeuralDWAV, self).__init__()
        
        if Input_Level > 8 and Input_Archi == "WPT":
            raise ValueError("Level should not be too high with WPT architecture")
        
        self.Size = Input_Size
        self.level = Input_Level
        self.Archi = Input_Archi
        
        self.lossL1 = nn.L1Loss(reduction='mean')
        
        #Generate the Filter class
        self.Filt = Filter(Size=Input_Size,
                           Level=Input_Level,
                           Archi=Input_Archi,
                           Filt_Trans=Filt_Trans,
                           Filt_Train=Filt_Train,
                           Filt_Tfree=Filt_Tfree,
                           Filt_Style=Filt_Style,
                           Filt_Mother=Filt_Mother)

        #Generate the Activation class
        self.Act = Activation(Level=Input_Level,
                              Archi=Input_Archi,
                              Act_Train=Act_Train,
                              Act_Style=Act_Style,
                              Act_Symmetric=Act_Symmetric,
                              Act_Init=Act_Init)
           
        
        #Make the transforms 
        if self.Archi == "WPT":
            self.T = self.LWPT
        elif self.Archi == "DWT": 
            self.T = self.LDWT
        else:
            raise ValueError("Bad input name")
        #Make the inverse transofrmation
        if Filt_Trans:
            if self.Archi == "WPT":
                self.iT = self.iLWPT
            elif self.Archi == "DWT":
                self.iT = self.iLDWT
            else:
                raise ValueError("Bad input name")
    

    def LWPT(self,x):
        Embeddings = [[]] * 2 ** self.level
        Embeddings[0] = x
        for i in range(self.level):
            ind_hp = 0
            ind_pr = 0
            for j in range(int(2**i)):
                ind_lp = int(ind_hp)
                ind_hp = int(ind_lp + (2**self.level)/(2**(i+1)))
                Embeddings[ind_hp] = self.Act(
                                        self.Filt(Embeddings[ind_pr],i+1,2*j+1),
                                     i+1,2*j+1)
                Embeddings[ind_lp] = self.Act(
                                        self.Filt(Embeddings[ind_pr],i+1,2*j),
                                     i+1,2*j)
                ind_pr = int(ind_pr + (2**self.level)/(2**i))
                ind_hp = int(ind_hp + (2**self.level)/(2**(i+1)))
        return Embeddings
                
    def iLWPT(self,Embeddings):
        for i in range(self.level-1, -1, -1):
            ind_hp = 0
            ind_pr = 0
            for j in range(int(2**i)):
                ind_lp = int(ind_hp)
                ind_hp = int(ind_lp + (2**self.level)/(2**(i+1)))
                Embeddings[ind_pr] = self.Filt.iforward(
                            Embeddings[ind_lp],Embeddings[ind_hp],i+1,2*j
                            )
                ind_pr = int(ind_pr + (2**self.level)/(2**i))
                ind_hp = int(ind_hp + (2**self.level)/(2**(i+1)))
        return Embeddings[0]
     

    def LDWT(self,x):
        Embeddings = [[]] * (self.level+1)
        for i in range(self.level):
            Embeddings[i] = self.Act(self.Filt(x,i+1,1),
                                 i+1,1)
            x = self.Act(self.Filt(x,i+1,0),
                                 i+1,0)
        Embeddings[self.level] = x
        return Embeddings        
    
    def iLDWT(self,Embeddings):
        x=Embeddings[self.level]
        for i in range(self.level-1, -1, -1):
            x = self.Filt.iforward(x,Embeddings[i],i+1,0)
        return x   

    def forward(self,x):
        return self.iT(self.T(x))
    
    def Reduce(self,emb):
        if self.Archi == "WPT":
            x=torch.abs(torch.stack(emb, dim=1)).squeeze(2)
            x=torch.mean(x,2)
        elif self.Archi == "DWT":
            x = []
            for i in range(self.level+1):
                x.append(torch.mean(torch.abs(emb[i]),2).squeeze())
            x = torch.stack(x, dim=1)
        return x

    
    def Embedding(self,emb):
        if self.Archi == "WPT":
            x=torch.abs(torch.stack(emb, dim=1)).squeeze(2)
        elif self.Archi == "DWT":
            NP=2**8
            x=torch.zeros(self.level+1,NP)
            for i in range(self.level+1):
                X=emb[i][0][0]
                v = torch.floor(torch.linspace(0,X.size(0),NP+1))
                for k in range(NP):
                    x[i,k]=torch.mean(torch.abs(X[int(v[k]):int((v[k+1])+1)]))
        return x
    
    def L1_sum(self,emb):
        if self.Archi == "WPT":
            x=torch.abs(torch.stack(emb, dim=1)).squeeze(2)
            x=self.lossL1(x,torch.zeros_like(x))
        elif self.Archi == "DWT":
            x=torch.cat(emb,dim=2)
            x=self.lossL1(x,torch.zeros_like(x))
        return x
      
        
class Filter(nn.Module):
    def __init__(self,
                 Size,
                 Level = 5,
                 Archi = "WPT",
                 Filt_Trans = True,
                 Filt_Train = True,
                 Filt_Tfree = False,
                 Filt_Style = "Filter_Free",
                 Filt_Mother = "db4"):
        super(Filter, self).__init__()
        self.inputSize = Size
        self.level = Level
        
        kernelInit = pywt.Wavelet(Filt_Mother).filter_bank[0]
        kernelInit = np.array(kernelInit)[np.newaxis, np.newaxis, :].copy()
        self.lK = len(kernelInit[0][0])  # Number of kernel parameter
        self.cmf = torch.tensor(np.array([(-1)**(i) for i in range(self.lK)])[np.newaxis, np.newaxis, :])
        self.pad = (self.lK//2-1, self.lK//2)        
        self.kernel, self.position, self.KerFun = self.Kernel_gen(
                            Archi = Archi,
                            kernelInit=kernelInit,
                            kernTrainable=Filt_Train,
                            Style=Filt_Style)  

        if Filt_Trans == True:
            self.mask, self.ipad, self.layerSize = self.GetTransposeInfo()
            if Filt_Tfree:
                self.kernelT, _, self.KerFunT = self.Kernel_gen(
                                                Archi = Archi,
                                                kernelInit=kernelInit,
                                                kernTrainable=True,
                                                Style=Filt_Style,
                                                transpose=True)
            else:
                self.kernelT = self.kernel
                self.KerFunT = [[]]
                for i in range(self.level):
                    self.KerFunT.append(list(np.array(self.KerFun[i+1])+2))

    def Kernel_gen(self, Archi, kernelInit, kernTrainable, Style, transpose=False):
        # Initialisation based on conjugate mirror filters properties
        if transpose == False:
            K_lp = torch.tensor(kernelInit)
            K_hp = torch.multiply(torch.flip(torch.tensor(kernelInit), [2]),
                self.cmf)
        else:
            K_lp = torch.flip(torch.tensor(kernelInit), [2])
            K_hp = torch.multiply(torch.tensor(kernelInit),
                                  self.cmf*-1)
        # Generate one learnable kernels for each filter
        kernel = []
        Pos = [[]]
        KerFun = [[]]
        c = 0
        #"All_Free", #or "Filter_Free", "Module_Free", "Layer_Free", "Kernel_Free"
        if Archi == "WPT":
            if Style == "Filter_Free":
                for i in range(self.level):
                    Pos.append([])
                    KerFun.append([])
                    for j in range(int(2**i)):
                        if (j % 2) == 0:
                            kernel.append(nn.Parameter(
                                data=K_lp.clone(),
                                requires_grad=kernTrainable))
                            kernel.append(nn.Parameter(
                                data=K_hp.clone(),
                                requires_grad=kernTrainable))
                        else:
                            kernel.append(nn.Parameter(
                                data=K_hp.clone(),
                                requires_grad=kernTrainable))
                            kernel.append(nn.Parameter(
                                data=K_lp.clone(),
                                requires_grad=kernTrainable))
                        Pos[i+1].append(c)
                        c += 1
                        Pos[i+1].append(c)
                        c += 1
                        KerFun[i+1].append(0)
                        KerFun[i+1].append(0)
            elif Style == "Module_Free":
                for i in range(self.level):
                    Pos.append([])
                    KerFun.append([])
                    for j in range(int(2**i)):
                        if (j % 2) == 0:
                            Ker = nn.Parameter(
                                data=K_lp.clone(),
                                requires_grad=kernTrainable)
                            kernel.append(Ker)
                            kernel.append(Ker)
                            KerFun[i+1].append(0)
                            if transpose:
                                KerFun[i+1].append(4)
                            else:
                                KerFun[i+1].append(1)
                        else:
                            Ker = nn.Parameter(
                                data=K_lp.clone(),
                                requires_grad=kernTrainable)
                            kernel.append(Ker)
                            kernel.append(Ker)
                            if transpose:
                                KerFun[i+1].append(4)
                            else:
                                KerFun[i+1].append(1)
                            KerFun[i+1].append(0)
                        Pos[i+1].append(c)
                        Pos[i+1].append(c)
                        c += 1

            elif Style == "Layer_Free":
                for i in range(self.level):
                    Pos.append([])
                    KerFun.append([])
                    Ker = nn.Parameter(
                        data=K_lp.clone(),
                        requires_grad=kernTrainable)
                    for j in range(int(2**i)):
                        if (j % 2) == 0:
                            kernel.append(Ker)
                            kernel.append(Ker)
                            KerFun[i+1].append(0)
                            if transpose:
                                KerFun[i+1].append(4)
                            else:
                                KerFun[i+1].append(1)
                        else:
                            kernel.append(Ker)
                            kernel.append(Ker)
                            if transpose:
                                KerFun[i+1].append(4)
                            else:
                                KerFun[i+1].append(1)
                            KerFun[i+1].append(0)
                        Pos[i+1].append(c)
                        Pos[i+1].append(c)
                    c += 1
            elif Style == "Kernel_Free":
                Ker = nn.Parameter(
                    data=K_lp.clone(),
                    requires_grad=kernTrainable)
                for i in range(self.level):
                    Pos.append([])
                    KerFun.append([])
                    for j in range(int(2**i)):
                        if (j % 2) == 0:
                            kernel.append(Ker)
                            kernel.append(Ker)
                            KerFun[i+1].append(0)
                            if transpose:
                                KerFun[i+1].append(4)
                            else:
                                KerFun[i+1].append(1)
                        else:
                            kernel.append(Ker)
                            kernel.append(Ker)
                            if transpose:
                                KerFun[i+1].append(4)
                            else:
                                KerFun[i+1].append(1)
                            KerFun[i+1].append(0)
                        Pos[i+1].append(c)
                        Pos[i+1].append(c)
            else:
                raise ValueError("Bad kernel style name")    
        elif Archi == "DWT":
            if Style == "Filter_Free":
                for i in range(self.level):
                    Pos.append([])
                    KerFun.append([])
                    kernel.append(nn.Parameter(
                        data=K_lp.clone(),
                        requires_grad=kernTrainable))
                    kernel.append(nn.Parameter(
                        data=K_hp.clone(),
                        requires_grad=kernTrainable))
                    Pos[i+1].append(c)
                    c += 1
                    Pos[i+1].append(c)
                    c += 1
                    KerFun[i+1].append(0)
                    KerFun[i+1].append(0)
            elif Style == "Module_Free" or Style == "Layer_Free":
                for i in range(self.level):
                    Pos.append([])
                    KerFun.append([])
                    Ker = nn.Parameter(
                        data=K_lp.clone(),
                        requires_grad=kernTrainable)
                    kernel.append(Ker)
                    kernel.append(Ker)
                    Pos[i+1].append(c)
                    Pos[i+1].append(c)
                    c += 1
                    KerFun[i+1].append(0)
                    if transpose:
                        KerFun[i+1].append(4)
                    else:
                        KerFun[i+1].append(1)
            elif Style == "Kernel_Free":
                Ker = nn.Parameter(
                    data=K_lp.clone(),
                    requires_grad=kernTrainable)
                for i in range(self.level):
                    Pos.append([])
                    KerFun.append([])
                    kernel.append(Ker)
                    kernel.append(Ker)
                    KerFun[i+1].append(0)
                    if transpose:
                        KerFun[i+1].append(4)
                    else:
                        KerFun[i+1].append(1)
                    Pos[i+1].append(c)
                    Pos[i+1].append(c)
            else:
                raise ValueError("Bad kernel style name") 
        else:
            raise ValueError("Bad architecture name")  

        return nn.ParameterList(kernel), Pos, KerFun

    def GetTransposeInfo(self):
        mask = []
        pad_iWPT = []
        layerSize = [(self.inputSize // (2**i)) + 1*(self.inputSize % (2**i) > 0)
                          for i in range(self.level+1)]  # Predict even/odd at each layers
        for i in range(self.level):
            pre_mask = [False, True]*layerSize[i+1]
            if layerSize[i] % 2 == 0:
                pad_iWPT.append(self.pad)
            else:
                pre_mask = pre_mask[1:]
                pad_iWPT.append((self.lK//2, self.lK//2-1))
            mask.append(torch.tensor(np.array(pre_mask)))
        return mask, pad_iWPT, layerSize

    #operation on kernel
    def fun(self,k,n):
        if n==0:
            return k
        elif n==1:
            return torch.multiply(torch.flip(k,[2]),self.cmf)
        elif n==2:
            return torch.flip(k,[2])
        elif n==3:
            return torch.multiply(k,self.cmf*-1)  
        elif n==4:
            return self.fun(self.fun(k,2),3) 
        else:
            raise ValueError("n value in fun Octopus.Filter.fun too high")

    #upsampling operation
    def Up_op(self, x, i_lvl):
        UpSig = torch.zeros(
            x.size(dim=0), 1, self.layerSize[i_lvl], dtype=torch.float64)
        for i_batch in range(x.size(dim=0)):
            UpSig[i_batch][0][self.mask[i_lvl]] = x[i_batch][0]
        return UpSig

    #WPT filter     
    def forward(self,x,curr_level,node):
        return F.conv1d(F.pad(x, 
                              pad=self.pad, 
                              mode='constant'),
                        self.fun(self.kernel[self.position[curr_level][node]],
                                  self.KerFun[curr_level][node]),
                        stride=2) 
    
    #inverse WPT operation  
    def iforward(self,x1,x2,curr_level,node):
        T1 = F.conv1d(F.pad(
                        self.Up_op(x1,curr_level-1),
                        pad=self.ipad[curr_level-1],
                        mode='constant'),
                    self.fun(self.kernelT[self.position[curr_level][node]],
                             self.KerFunT[curr_level][node]),
                    stride=1)
        T2 = F.conv1d(F.pad(
                        self.Up_op(x2,curr_level-1),
                        pad=self.ipad[curr_level-1],
                        mode='constant'),
                    self.fun(self.kernelT[self.position[curr_level][node+1]],
                             self.KerFunT[curr_level][node+1]),
                    stride=1)
        return  torch.add(T1, T2)    
        
class Activation(nn.Module):
    def __init__(self,
                 Level,
                 Archi,
                 Act_Train=True,
                 Act_Style="Sigmoid",
                 Act_Symmetric=True,
                 Act_Init=0):
        super(Activation, self).__init__()
        self.level = Level
        
        self.bias_p, self.position = self.Biases_gen(Archi,Act_Init,Act_Train)
        if Act_Symmetric:
            self.bias_n = self.bias_p
        else:
            self.bias_n, _ = self.Biases_gen(Archi,Act_Init,Act_Train)
    
        # Define function
        if Act_Style == "Sigmoid":
            self.HT = nn.Sigmoid()
            self.forward = self.Thresh_SigSym
        elif Act_Style == "Relu":
            self.forward = self.Thresh_ReluSym
        ###Create the activation function you want
        else:
            raise ValueError("Bad Activation function name")

    def Biases_gen(self, Archi, initHT, trainHT):
        # Generate one learnable bias for each filter
        biases = []
        Pos = [[]]
        c = 0
        if Archi == "WPT":
            for i in range(self.level):
                Pos.append([])
                for j in range(int(2**i)):
                    biases.append(nn.Parameter(
                        data=torch.tensor(
                            np.array([initHT])[np.newaxis, np.newaxis, :].copy(), dtype=torch.float64).clone(),
                        requires_grad=trainHT))
                    biases.append(nn.Parameter(
                        data=torch.tensor(
                            np.array([initHT])[np.newaxis, np.newaxis, :].copy(), dtype=torch.float64).clone(),
                        requires_grad=trainHT))
                    Pos[i+1].append(c)
                    c += 1
                    Pos[i+1].append(c)
                    c += 1
        elif Archi == "DWT":
            for i in range(self.level):
                Pos.append([])
                biases.append(nn.Parameter(
                    data=torch.tensor(
                        np.array([initHT])[np.newaxis, :].copy(), dtype=torch.float64).clone(),
                    requires_grad=trainHT))
                biases.append(nn.Parameter(
                    data=torch.tensor(
                        np.array([initHT])[np.newaxis, :].copy(), dtype=torch.float64).clone(),
                    requires_grad=trainHT))
                Pos[i+1].append(c)
                c += 1
                Pos[i+1].append(c)
                c += 1            
        else:
            raise ValueError("Bad architecture name")
        return nn.ParameterList(biases), Pos

        
    def Thresh_SigSym(self, x, curr_level, node):
        return torch.multiply(x, 
                    self.HT(10*(x-self.bias_p[self.position[curr_level][node]]))+self.HT(-10*(x+self.bias_n[self.position[curr_level][node]])))
    def Thresh_ReluSym(self, x, curr_level, node):
        return F.relu(x-self.bias_p[self.position[curr_level][node]])-F.relu(-x-self.bias_n[self.position[curr_level][node]])


