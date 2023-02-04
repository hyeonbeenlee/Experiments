import matplotlib.pyplot as plt
import pandas as pd
import torch
import Functions as Func
import glob
import numpy as np
from torch.utils.data import TensorDataset
np.random.seed(777)
torch.manual_seed(777)



#변환할 파일 리스트들
DataList=glob.glob('MBD Data/Pendulum_Single_Full/*.csv')
#함수로 랜덤스플릿 실행
Train_DataList,Valid_DataList=Func.SplitData(DataList,0.8,0.2)

#데이터프레임으로 병합
TrainData=pd.DataFrame()
ValidData=pd.DataFrame()
for datapath in Train_DataList:
    Data=pd.read_csv(datapath)
    TrainData=pd.concat([TrainData,Data],ignore_index=True)
for datapath in Valid_DataList:
    Data=pd.read_csv(datapath)
    ValidData=pd.concat([ValidData,Data],ignore_index=True)

Model=Func.DNNModel()
Model.UseTrainData(TrainData) #모델 클래스에 학습데이터 입력해줌(정규화용)

HiddenLs=3
Nodes=256
BatchSize=256
Epochs=100
inD=4
outD=3
LearnRate=1e-5
Input=['Time','L','c','th_0']
Output=['th','dth','ddth']

Model.SetInputOutput(Input,Output)
Model.SetModel(inD,Nodes,outD,HiddenLs)
Optimizer=torch.optim.Adam(Model.parameters(),lr=LearnRate)
ModelFile=torch.load('PendulumSingleFull.pt')
Model.load_state_dict(ModelFile)
Model.eval()
Model.cuda()

Data=pd.read_csv('MBD Data/Pendulum_Single_Full\\L=0.11,c=0.09,th_0=-0.63.csv')
Input=['Time','L','c','th_0']
Output=['th','dth','ddth']
TestX,Label=Data[Input],Data[Output]
TestX=Model.NormalizeInput(TestX)
TestX=torch.FloatTensor(TestX.to_numpy()).cuda()
Prediction=torch.zeros(Label.shape).cuda()

with torch.no_grad():
    for i in range(len(TestX)):
        Prediction[i,:]=Model(TestX[i,:])


    Prediction=np.array(Prediction.cpu())
    Prediction=pd.DataFrame(Prediction,columns=Output)
    Prediction=Model.InverseOutput(Prediction)


plt.subplot(311)
plt.plot(Data['Time'],Label['th'])
plt.plot(Data['Time'],Prediction['th'])
plt.grid()
plt.subplot(312)
plt.plot(Data['Time'],Label['dth'])
plt.plot(Data['Time'],Prediction['dth'])
plt.grid()
plt.subplot(313)
plt.plot(Data['Time'],Label['ddth'])
plt.plot(Data['Time'],Prediction['ddth'])
plt.grid()
plt.show()