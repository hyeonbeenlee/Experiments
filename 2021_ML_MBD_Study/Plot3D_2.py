import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import Functions as Func
import torch

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

DataList=glob.glob('MBD Data/Pendulum_Single/Pendulum_Single_Full/*.csv') #테스트용
nK=Func.Loader('MBD Data/Pendulum_Single/2K.csv') #정규화 기준
ModelFilePath=f'DNN Model/Pendulum_Single/2K.pt'

DataList=DataList[:11]


for datapath in DataList:
    data=pd.read_csv(datapath,header=None).T
    data.rename(columns=data.iloc[0],inplace=True)
    data.drop(0,axis=0,inplace=True)
    data=nK.Normalize(data)
    ax.plot(data['Time'],data['th_0'],data['th'],c='r')


inD=4
outD=3
HiddenLayers=3
Nodes=256
BatchSize=256
Epochs=100
LearningRate=1e-4
DoShuffle=True
torch.manual_seed(777)

Model=DNNModel(inD,Nodes,outD,HiddenLayers).cuda()
ModelFile=torch.load(ModelFilePath)
Model.load_state_dict(ModelFile)
Model.eval()

THSurf=np.empty((201,201))
Time=np.linspace(0,1,201,endpoint=True)
L=np.full(201,0)
C=np.full(201,0)

for idx,th0 in enumerate(np.linspace(0,1,201,endpoint=True)):
    TH_0s=np.full(201,th0)
    Input=pd.DataFrame({'Time':Time,'L':L,'c':C,'th_0':TH_0s}).to_numpy()
    Input=torch.FloatTensor(Input).cuda()
    with torch.no_grad():
        Output = Model(Input).cpu()
        THSurf[idx]=Output[:,0]

print(THSurf)
TH_0s=np.linspace(0,1,201,endpoint=True)
Time,TH_0s=np.meshgrid(Time,TH_0s)
print(Time)
print(TH_0s)

ax.plot_surface(Time,TH_0s,THSurf)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])
ax.set_xlabel('Time')
ax.set_ylabel(r'$\theta_0(rad)$')
ax.set_zlabel(r'$\theta(rad)$')

plt.show()