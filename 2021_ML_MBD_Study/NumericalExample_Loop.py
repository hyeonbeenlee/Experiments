import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions as Func
import torch
import torch.nn.functional as nnF
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score as R2
from scipy.signal import argrelextrema
import time

def Numerical_1(x):
    y=np.sin(x)**3+np.cos(x)**3 #[0,7]
    return y

def Numerical_2(x):
    y=x*np.sin(x)+x*np.cos(2*x) #[0,10]
    return y

def Numerical_3(x):
    y=0
    for k in range(6):
        y+=k*np.cos((k+1)*x+k) #[-10,10]
    return -y


ValidX1=np.linspace(0,7,1000)
ValidX2=np.linspace(0,10,1000)
ValidX3=np.linspace(-10,10,1000)
ValidY1=Numerical_1(ValidX1)
ValidY2=Numerical_2(ValidX2)
ValidY3=Numerical_3(ValidX3)


SampleNum=50
np.random.seed(777)
# TrainX1=np.sort(np.random.choice(ValidX1,size=SampleNum,replace=False))
# TrainX2=np.sort(np.random.choice(ValidX2,size=SampleNum,replace=False))
# TrainX3=np.sort(np.random.choice(ValidX3,size=SampleNum,replace=False))
TrainX1=np.linspace(0,7,SampleNum,endpoint=True)
TrainX2=np.linspace(0,10,SampleNum,endpoint=True)
TrainX3=np.linspace(-10,10,SampleNum,endpoint=True)
TrainY1=Numerical_1(TrainX1)
TrainY2=Numerical_2(TrainX2)
TrainY3=Numerical_3(TrainX3)



## 균일추출+극점 추가
PeakLoc1=np.concatenate([argrelextrema(ValidY1,np.greater),argrelextrema(ValidY1,np.less)],axis=None)
PeakLoc2=np.concatenate([argrelextrema(ValidY2,np.greater),argrelextrema(ValidY2,np.less)],axis=None)
PeakLoc3=np.concatenate([argrelextrema(ValidY3,np.greater),argrelextrema(ValidY3,np.less)],axis=None)
TrainX1=np.hstack([TrainX1,ValidX1[PeakLoc1]])
TrainY1=np.hstack([TrainY1,ValidY1[PeakLoc1]])
TrainX2=np.hstack([TrainX2,ValidX2[PeakLoc2]])
TrainY2=np.hstack([TrainY2,ValidY2[PeakLoc2]])
TrainX3=np.hstack([TrainX3,ValidX3[PeakLoc3]])
TrainY3=np.hstack([TrainY3,ValidY3[PeakLoc3]])
MaxLength=np.max([len(TrainX1),len(TrainX2),len(TrainX3)])

AdditionalX1=np.random.choice(TrainX1,size=MaxLength-len(TrainX1),replace=False)
AdditionalX2=np.random.choice(TrainX2,size=MaxLength-len(TrainX2),replace=False)
AdditionalY1=Numerical_1(AdditionalX1)
AdditionalY2=Numerical_2(AdditionalX2)

TrainX1=np.hstack([TrainX1,AdditionalX1])
TrainX2=np.hstack([TrainX2,AdditionalX2])
TrainY1=np.hstack([TrainY1,AdditionalY1])
TrainY2=np.hstack([TrainY2,AdditionalY2])

HP_HiddenLs=[3,4,5,6,7,8]
HP_Nodes=[16,32,64,128,256,512]
HP_Epochs=np.linspace(100,2000,4,endpoint=True,dtype=int)

for Alpha in HP_HiddenLs:
    for Beta in HP_Nodes:
        for Gamma in HP_Epochs:
            HiddenLs = Alpha
            Nodes = Beta
            Epochs = Gamma
            BatchSize = 128
            LearnRate = 1e-3
            DoShuffle = True
            Input=['x1','x2','x3']
            Output=['y1','y2','y3']

            inD, outD = len(Input), len(Output)

            # 데이터프레임으로 병합
            TrainData = pd.DataFrame({'x1':TrainX1,'x2':TrainX2,'x3':TrainX3,'y1':TrainY1,'y2':TrainY2, 'y3':TrainY3})
            ValidData = pd.DataFrame({'x1':ValidX1,'x2':ValidX2,'x3':ValidX3,'y1':ValidY1,'y2':ValidY2,'y3':ValidY3})

            Model = Func.DNNModel()
            Model.UseTrainData(TrainData)
            Model.SetInputOutput(Input, Output)
            Model.SetModel(inD, Nodes, outD, HiddenLs)
            Optimizer=torch.optim.Adam(lr=LearnRate,params=Model.parameters())

            # 입출력 나눔
            TrainX, TrainY = TrainData[Input], TrainData[Output]
            ValidX, ValidY = ValidData[Input], ValidData[Output]

            # 정규화
            TrainX, TrainY = Model.NormalizeInput(TrainX), Model.NormalizeOutput(TrainY)
            ValidX, ValidY = Model.NormalizeInput(ValidX), Model.NormalizeOutput(ValidY)

            # 넘파이 변환
            TrainX, TrainY = TrainX.to_numpy(), TrainY.to_numpy()
            ValidX, ValidY = ValidX.to_numpy(), ValidY.to_numpy()

            # 텐서 변환, GPU전송
            TrainX, TrainY = torch.FloatTensor(TrainX).cuda(), torch.FloatTensor(TrainY).cuda()
            ValidX, ValidY = torch.FloatTensor(ValidX).cuda(), torch.FloatTensor(ValidY).cuda()
            Model.cuda()

            # 텐서 데이터셋으로 묶음
            TrainSet = TensorDataset(TrainX, TrainY)
            ValidSet = TensorDataset(ValidX, ValidY)

            # 텐서 데이터로더
            Train_DLoader = DataLoader(TrainSet, batch_size=BatchSize, shuffle=True)

            ## 학습 코드
            Start = time.time()
            for i in range(Epochs):
                for batchidx, batchdata in enumerate(Train_DLoader):
                    trainX, trainY = batchdata  # 입출력 데이터 선언
                    Pred = Model(trainX)  # 출력
                    Loss = nnF.mse_loss(Pred, trainY)  # 손실계산
                    Optimizer.zero_grad()  # Autograd 초기화
                    Loss.backward()  # 역전파
                    Optimizer.step()  # 가중치 수정

                # print(f"{i + 1}/{Epochs}, Loss={Loss.item():.6f}")

            End = time.time()
            Time = int(End - Start)

            ## 최종 결정계수 평가
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

            # torch.save(Model.state_dict(), f"DNN Model/{Example}/{Case}.pt")  # 모델 파라미터 저장

            print(f"Hidden layers={Alpha}, Nodes={Beta},Epochs={Gamma}")
            print(f"Batch size={BatchSize}, Learning rate={LearnRate}")
            print(f"Train data size: {len(TrainData)}")
            print(f"Learning time {Time}sec.\n")
            # print(f'Example-{Example}, Case-{Case} Model Saved.')