import torch
import torch.nn.functional as nnF
from torch.utils.data import TensorDataset, DataLoader, random_split
import Functions as Func
import time
import glob
from sklearn.metrics import r2_score as R2
import pandas as pd
import numpy as np

np.random.seed(777)
torch.manual_seed(777)

## 모델 변수 선언
Example = 'Pendulum_Single'
Case = '2K+5th_0'
ModelFile=torch.load('DNN Model/Pendulum_Single/2K.pt') # 추가 학습시킬 모델파일

if Example == 'Pendulum_Single':
    HiddenLs = 3
    Nodes = 256
    BatchSize = 256
    Epochs = 100

    LearnRate = 1e-4
    DoShuffle = True
    Input = ['Time', 'L', 'c', 'th_0']
    Output = ['th', 'dth', 'ddth']
    AppendixPath = 'MBD Data/Pendulum_Single/Appendix_Pendulum_Single.csv'
    Train_DList = glob.glob(f"MBD Data/Pendulum_Single/Pendulum_Single_{Case}/*.csv")
    ValidDataPath = 'MBD Data/Pendulum_Single/ValidData.csv'

elif Example == 'Pendulum_Double':
    HiddenLs = 4
    Nodes = 64
    BatchSize = 1024
    Epochs = 400
    LearnRate = 1e-4
    DoShuffle = True
    Input = ['Time', 'L1', 'L2', 'dth1_0', 'dth2_0']
    Output = ['th1', 'th2', 'dth1', 'dth2']
    AppendixPath = 'MBD Data/Pendulum_Double/Appendix_Pendulum_Double.csv'
    Train_DList = glob.glob(f"MBD Data/Pendulum_Double/Pendulum_Double_{Case}/*.csv")
    ValidDataPath = 'MBD Data/Pendulum_Double/ValidData.csv'

elif Example == 'SliderCrankKin':
    HiddenLs = 2
    Nodes = 128
    BatchSize = 64
    Epochs = 200
    LearnRate = 1e-4
    Input = ['Time', 'tau', 'r', 'L/r']
    Output = ['th', 'pi', 'dpi', 'ddpi', 'xb', 'dxb', 'ddxb']
    AppendixPath = 'MBD Data/SliderCrankKin/Appendix_SliderCrankKin.csv'
    Train_DList = glob.glob(f"MBD Data/SliderCrankKin/SliderCrank_{Case}/*.csv")
    ValidDataPath = 'MBD Data/SliderCrankKin/ValidData.csv'

inD, outD = len(Input), len(Output)

# 데이터프레임으로 병합
TrainData = pd.DataFrame()
ValidData = Func.CollectConcat('MBD Data/Pendulum_Single/TestData1')
for datapath in Train_DList:
    Data = pd.read_csv(datapath)
    TrainData = pd.concat([TrainData, Data], ignore_index=True)

Model = Func.DNNModel()
Model.UseTrainData(TrainData)
Model.SetInputOutput(Input, Output)
Model.SetModel(inD, Nodes, outD, HiddenLs)
Optimizer = torch.optim.Adam(Model.parameters(), lr=LearnRate)
Model.load_state_dict(ModelFile) #2K 모델의 파라미터를 불러온다
Model.train()

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

    print(f"{i + 1}/{Epochs}, Loss={Loss.item():.6f}")

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

torch.save(Model.state_dict(), f"DNN Model/{Example}/{Case}.pt")  # 모델 파라미터 저장
End = time.time()
Time = int(End - Start)

print(f"Learning time {Time}sec.")
print(f'Example-{Example}, Case-{Case} Model Saved.')
