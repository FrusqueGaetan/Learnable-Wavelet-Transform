import sys
sys.path.append('Spyder_Part1/NeuralDWAV/Code/')
import NeuralDWAV
import Util_NeuralDWAV as Utils
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Np=2**13#Number of sample in the signals
Generator = Utils.Generator_test(Np)
LDWT = NeuralDWAV.NeuralDWAV(Np,
                              Input_Level=8,#WPT with 5 level resulting to 32 outputs
                              Input_Archi="DWT")


LDWT.to(device)
loss_MSE = torch.nn.MSELoss()
optimizer = torch.optim.Adam(LDWT.parameters(),
                              lr=0.001,betas=(0.9,0.999),eps=1e-7)
x_test, y_test=Generator. __getitem__(1)
x_est0 = LDWT(x_test)
Emb0 = LDWT.T(x_test)
BS=8#Works better with low batch size
H=[]
for Epoch in range(1000):
    X,Y=Generator. __getitem__(BS)
    LDWT.zero_grad()
    loss=loss_MSE(LDWT(X),Y)
    loss.backward()#
    optimizer.step()#
    H.append(loss.item())
LDWT.eval()
x_est=LDWT(x_test)


Utils.plt_x(1,[x_test,x_est0,x_est],["Test signal","Reconstruction WPT", "Reconstruction LWPT"])
Utils.plt_Emb(2,[LDWT.Embedding(Emb0),LDWT.Embedding(LDWT.T(x_est0))],["WPT representation","LWPT representation"])




#%%Using Unsupervised lossLWPT.to(device)
Np=2**13#Number of sample in the signals
Generator = Utils.Generator_test(Np)
DESPAWN = NeuralDWAV.NeuralDWAV(Np,
                              Input_Level=8,#WPT with 5 level resulting to 32 outputs
                              Input_Archi="DWT")
loss_L1 = torch.nn.L1Loss()
optimizer = torch.optim.Adam(DESPAWN.parameters(),
                              lr=0.01,betas=(0.9,0.999),eps=1e-7)
x_test, y_test=Generator. __getitem__(1)
x_est0 = DESPAWN(x_test)
Emb0 = DESPAWN.T(x_test)
BS=8#Works better with low batch size
Lambda=1
H=[]
for Epoch in range(1000):
    X,Y=Generator. __getitem__(BS)
    DESPAWN.zero_grad()
    Emb=DESPAWN.T(X)
    loss=loss_L1(DESPAWN.iT(Emb.copy()),X)+Lambda*DESPAWN.L1_sum(Emb)
    loss.backward()#
    optimizer.step()#
    H.append(loss.item())
DESPAWN.eval()
x_est=DESPAWN(x_test)


#Utils.plt_x(3,[x_test,x_est0,x_est],["Test signal","Reconstruction WPT", "Reconstruction LWPT"])
Utils.plt_Emb(4,[DESPAWN.Embedding(Emb0),DESPAWN.Embedding(DESPAWN.T(x_est0))],["WPT representation","LWPT representation"])

