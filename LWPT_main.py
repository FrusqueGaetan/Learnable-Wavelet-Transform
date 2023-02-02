import sys
sys.path.append('Spyder_Part1/NeuralDWAV/Code/')
import NeuralDWAV
import Util_NeuralDWAV as Utils
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Np=2**13#Number of sample in the signals
Generator = Utils.Generator_test(Np)
LWPT = NeuralDWAV.NeuralDWAV(Np,
                              Input_Level=5,#WPT with 5 level resulting to 32 outputs
                              Input_Archi="WPT")


LWPT.to(device)
loss_MSE = torch.nn.MSELoss()
optimizer = torch.optim.Adam(LWPT.parameters(),lr=0.001)
x_test, y_test=Generator. __getitem__(1)
x_est0 = LWPT(x_test)
Emb0 = LWPT.T(x_test)
BS=8#Works better with low batch size
H=[]
for Epoch in range(1000):
    X,Y=Generator. __getitem__(BS)
    LWPT.zero_grad()
    loss=loss_MSE(LWPT(X),Y)
    loss.backward()#
    optimizer.step()#
    H.append(loss.item())
LWPT.eval()
x_est=LWPT(x_test)


Utils.plt_x(1,[x_test,x_est0,x_est],["Test signal","Reconstruction WPT", "Reconstruction LWPT"])
Utils.plt_Emb(2,[LWPT.Embedding(Emb0),LWPT.Embedding(LWPT.T(x_est0))],["WPT representation","LWPT representation"])




#%%Using Unsupervised loss from DESPAWN, denoising is not the 1st goal of this losss, but more signal representation
Np=2**13#Number of sample in the signals
Generator = Utils.Generator_test(Np)
LWPT = NeuralDWAV.NeuralDWAV(Np,
                              Input_Level=5,#WPT with 5 level resulting to 32 outputs
                              Input_Archi="WPT")
loss_L1 = torch.nn.L1Loss()
optimizer = torch.optim.Adam(LWPT.parameters(),
                              lr=0.01,betas=(0.9,0.999),eps=1e-7)
x_test, y_test=Generator. __getitem__(1)
x_est0 = LWPT(x_test)
Emb0 = LWPT.T(x_test)
BS=8#Works better with low batch size
Lambda=1
H=[]
for Epoch in range(1000):
    X,Y=Generator. __getitem__(BS)
    LWPT.zero_grad()
    Emb=LWPT.T(X)
    loss=loss_L1(LWPT.iT(Emb.copy()),X)+Lambda*LWPT.L1_sum(Emb)
    loss.backward()#
    optimizer.step()#
    H.append(loss.item())
LWPT.eval()
x_est=LWPT(x_test)


#Utils.plt_x(3,[x_test,x_est0,x_est],["Test signal","Reconstruction WPT", "Reconstruction LWPT"])
Utils.plt_Emb(4,[LWPT.Embedding(Emb0),LWPT.Embedding(LWPT.T(x_est0))],["WPT representation","LWPT representation"])

