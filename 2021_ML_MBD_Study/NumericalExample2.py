import matplotlib.markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions as Func
import torch
import torch.nn.functional as nnF
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE
from scipy.signal import argrelextrema
import time
np.random.seed(777)
torch.manual_seed(777)

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

## Time dimension 데이터 축소를 위한 컴퓨터 실험


ValidX1=np.linspace(-10,10,1000)
# ValidX2=np.linspace(0,10,1000)
# ValidX3=np.linspace(-10,10,1000)
ValidY1=Numerical_3(ValidX1)
# ValidY2=Numerical_2(ValidX2)
# ValidY3=Numerical_3(ValidX3)


SampleNum=50
# TrainX1=np.sort(np.random.choice(ValidX1,size=SampleNum,replace=False))
TrainX1=np.linspace(-10,10,SampleNum,endpoint=True)
TrainY1=Numerical_3(TrainX1)




# ## 균일추출+극점 추가
PeakLoc1=np.concatenate([argrelextrema(ValidY1,np.greater),argrelextrema(ValidY1,np.less)],axis=None)
TrainX1=np.hstack([TrainX1,ValidX1[PeakLoc1]])
TrainY1=np.hstack([TrainY1,ValidY1[PeakLoc1]])
MaxLength=np.max([len(TrainX1)])

AdditionalX1=np.random.choice(TrainX1,size=MaxLength-len(TrainX1),replace=False)
AdditionalY1=Numerical_3(AdditionalX1)

TrainX1=np.hstack([TrainX1,AdditionalX1])
TrainY1=np.hstack([TrainY1,AdditionalY1])


## 모델 세팅
HiddenLs = 3
Nodes = 300
Epochs = 3000
BatchSize = 128
LearnRate = 1e-3
DoShuffle = True
Input=['x1']
Output=['y1']

inD, outD = len(Input), len(Output)

# 데이터프레임으로 병합
TrainData = pd.DataFrame({'x1':TrainX1,'y1':TrainY1})
ValidData = pd.DataFrame({'x1':ValidX1,'y1':ValidY1})

Model = Func.DNNModel()
Model.UseTrainData(TrainData)
Model.SetInputOutput(Input, Output)
Model.SetModel(input_dim=inD, nodes=Nodes, output_dim=outD, hidden_layers=HiddenLs,activation='relu')
Optimizer=torch.optim.Adam(lr=LearnRate,params=Model.parameters())

# 입출력 나눔
TrainX, TrainY = TrainData[Input], TrainData[Output]
ValidX, ValidY = ValidData[Input], ValidData[Output]

# 정규화
TrainX, TrainY = Model.NormalizeInput(TrainX,mode='gaussian'), Model.NormalizeOutput(TrainY,mode='gaussian')
ValidX, ValidY = Model.NormalizeInput(ValidX,mode='gaussian'), Model.NormalizeOutput(ValidY,mode='gaussian')

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
R2Scores=pd.DataFrame(np.zeros((Epochs,len(Output))),columns=Output)

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

    print(f"{i + 1}/{Epochs}, Loss={Loss.item():.6f}")
    # with torch.no_grad():
    #     Model.eval()
    #     Labels = torch.zeros(ValidY.shape).cuda()
    #     Predictions = torch.zeros(ValidY.shape).cuda()
    #
    #     for idx, ValidData in enumerate(ValidSet):
    #         validX, validY = ValidData
    #         Labels[idx, :] = validY
    #         PredY = Model.forward(validX)
    #         Predictions[idx, :] = PredY
    #
    #     Labels = pd.DataFrame(Labels.cpu(), columns=Output)
    #     Predictions = pd.DataFrame(Predictions.cpu(), columns=Output)
    #     Model.train()
    #
    #     for col in Output:
    #         R2Scores[col].iloc[i]=R2(Labels[col], Predictions[col])


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
Model.Summary(show_params=False)

# torch.save(Model.state_dict(), f"DNN Model/{Example}/{Case}.pt")  # 모델 파라미터 저장

print(f"Hidden layers={HiddenLs}, Nodes={Nodes},Epochs={Epochs}")
print(f"Batch size={BatchSize}, Learning rate={LearnRate}")
print(f"Train data size: {len(TrainData)}")
print(f"Learning time {Time}sec.\n")
# print(f'Example-{Example}, Case-{Case} Model Saved.')








ValidX=ValidX.cpu().numpy()
ValidX=pd.DataFrame(ValidX,columns=Input)
ValidX=Model.InverseInput(ValidX,mode='gaussian')
Labels,Predictions=Model.InverseOutput(Labels,mode='gaussian'),Model.InverseOutput(Predictions,mode='gaussian')





Func.MyPlotTemplate()

Line1=f"Hidden Layers={HiddenLs}, Nodes={Nodes}, Epochs={Epochs}, Learning time={Time}sec"
Line2=f"\nBatch Size={BatchSize}, Learning Rate={LearnRate}"
plt.suptitle(Line1+Line2)

plt.subplot(121)
plt.title(f"Test Function 1, R2={R2(Labels['y1'], Predictions['y1']):.6f}, MSE={MSE(Labels['y1'], Predictions['y1']):.6f}")
plt.plot(ValidX['x1'],Labels['y1'],c='b')
plt.plot(ValidX['x1'],Predictions['y1'],c='r')
plt.scatter(TrainX1,TrainY1,s=10,c='k',marker='o')
plt.legend(['Labels','Predictions','Train data points'])
plt.grid()

plt.subplot(122)
plt.plot(range(1,Epochs+1),R2Scores.iloc[:,0],c='y')
plt.xlim(1,Epochs);     plt.ylim(-1,1)
plt.xlabel('Epochs');   plt.ylabel(r'$R^2 Value$')
plt.grid()

plt.tight_layout()
plt.show()