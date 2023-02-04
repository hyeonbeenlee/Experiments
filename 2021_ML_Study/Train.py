import torch
import torch.nn.functional as nnF
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import glob
import Functions as Func
from sklearn.metrics import r2_score as R2
np.random.seed(777)
torch.manual_seed(777)

#변환할 파일 리스트들
DataList=glob.glob('MBD Data/Pendulum_Double_Full/*.csv')
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
print(TrainData)
print(ValidData)



Model=Func.DNNModel()
Model.UseTrainData(TrainData) #모델 클래스에 학습데이터 입력해줌(정규화용)

# HiddenLs=2
# Nodes=128
# BatchSize=64
# Epochs=200
# inD=4
# outD=7
# LearnRate=1e-4
# Input=['Time','tau','r','L/r']
# Output=['th','pi','dpi','ddpi','xb','dxb','ddxb']

# HiddenLs=3
# Nodes=256
# BatchSize=256
# Epochs=100
# inD=4
# outD=3
# LearnRate=1e-3 #개형이 간단하므로 학습률 높게 가져갑시다..
# Input=['Time','L','c','th_0']
# Output=['th','dth','ddth']

HiddenLs=4
Nodes=64
BatchSize=1024
Epochs=400
inD=5
outD=4
LearnRate=1e-4
Input=['Time','L1','L2','dth1_0','dth2_0']
Output=['th1','th2','dth1','dth2']

Model.SetInputOutput(Input,Output)
Model.SetModel(inD,Nodes,outD,HiddenLs)
Optimizer=torch.optim.Adam(Model.parameters(),lr=LearnRate)






#입출력 나눔
TrainX,TrainY=TrainData[Input],TrainData[Output]
ValidX,ValidY=ValidData[Input],ValidData[Output]

#정규화
TrainX,TrainY=Model.NormalizeInput(TrainX),Model.NormalizeOutput(TrainY)
ValidX,ValidY=Model.NormalizeInput(ValidX),Model.NormalizeOutput(ValidY)

#넘파이 변환
TrainX,TrainY=TrainX.to_numpy(),TrainY.to_numpy()
ValidX,ValidY=ValidX.to_numpy(),ValidY.to_numpy()

#텐서 변환, GPU전송
TrainX,TrainY=torch.FloatTensor(TrainX).cuda(),torch.FloatTensor(TrainY).cuda()
ValidX,ValidY=torch.FloatTensor(ValidX).cuda(),torch.FloatTensor(ValidY).cuda()
Model.cuda()

#텐서 데이터셋으로 묶음
TrainSet=TensorDataset(TrainX,TrainY)
ValidSet=TensorDataset(ValidX,ValidY)

#텐서 데이터로더
Train_DLoader=DataLoader(TrainSet,batch_size=BatchSize,shuffle=True)

for i in range(Epochs):
    for batchidx,batchdata in enumerate(Train_DLoader):
        trainX,trainY=batchdata #입출력 데이터 선언
        Pred=Model(trainX) #출력
        Loss=nnF.mse_loss(Pred,trainY) #손실계산
        Optimizer.zero_grad() #Autograd 초기화
        Loss.backward() #역전파
        Optimizer.step() #가중치 수정

    print(f"{i + 1}/{Epochs}, Loss={Loss.item():.6f}")

    if (i+1)%50==0:
        with torch.no_grad():
            Model.eval()
            Labels = torch.zeros(ValidY.shape).cuda()
            Predictions = torch.zeros(ValidY.shape).cuda()

            for idx, ValidData in enumerate(ValidSet):
                validX, validY = ValidData
                Labels[idx, :] = validY
                PredY = Model(validX)
                Predictions[idx, :] = PredY

            Labels = pd.DataFrame(Labels.cpu(), columns=Output)
            Predictions = pd.DataFrame(Predictions.cpu(), columns=Output)
            for col in Output:
                print(f"{col} R2: {R2(Labels[col], Predictions[col]):.6f}")
            Model.train()

torch.save(Model.state_dict(),"PendulumDoubleFull.pt")