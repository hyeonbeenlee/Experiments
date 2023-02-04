import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.utils.data import TensorDataset, DataLoader
import Functions as Func
import time
import glob
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np


def SetHyperParam():
    global BatchSize, Epochs, LearnRate, DoShuffle, UseLRScheduler, UseL2Regularizer, Input, Output, inD, outD
    np.random.seed(777)
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)
    BatchSize = 1000  # 256
    Epochs = 100000
    LearnRate = 1e-2
    DoShuffle = False
    UseLRScheduler = False
    UseL2Regularizer = False
    Input = [1,2,3,4,5]
    # Input=['Time', 'L', 'c', 'th_0']
    # Input=['Time', 'th_0']
    # Input = ['th', 'dth', 'ddth']
    # Input=['Time', 'th_0','th', 'dth', 'ddth']
    Output = Input
    inD, outD = len(Input), len(Output)


SetHyperParam()

## MBD 학습데이터 생성 및 병합
# MBD = Func.MBD_Integrator()
# DataFrames = []
# Count = 0
# K = 3
# Ls = np.linspace(0.1, 0.2, K, endpoint=True) # [0.1]  # np.linspace(0.1, 0.2, K, endpoint=True)
# cs = np.linspace(0, 0.15, K, endpoint=True) #[0]  # np.linspace(0, 0.15, K, endpoint=True)
# th_0s = np.linspace(-np.pi / 2, np.pi / 2, K, endpoint=True)
# # th_0s = np.sort(np.random.random(K)) * np.pi - np.pi / 2
# Train_Samples = th_0s
# for L in Ls:
#     for c in cs:
#         for th_0 in th_0s:
#             DataFrames.append(MBD.Pendulum_Single(L=L, c=c, th_0=th_0))
#             Count += 1
#             print(f"Generating MBD Train Data {Count}")
# TrainData = pd.concat(DataFrames, keys=range(1, len(DataFrames) + 1))
TrainData=pd.DataFrame(np.random.random((1000,inD)),columns=Input)
TrainData_Original = TrainData

## MBD 검증데이터 생성 및 병합
# ValidData = pd.concat(np.random.random(200,4))
ValidData = TrainData
ValidData_Original = ValidData

## 생성된 학습 및 검증 데이터 출력, 중복 검사
print(f"Raw Train Data\n{TrainData}\n")
# print(f"Valid Data\n{ValidData}\n")
# Warning = False
# for datapoint in Valid_Samples:
#     if datapoint in Train_Samples:
#         Warning = True
# if Warning:
#     print('학습데이터와 검증데이터가 분리되지 않음!')
# else:
#     print('학습데이터와 검증데이터가 분리됨')
# print(f"Train Data th_0s: \n{Train_Samples}")
# print(f"Valid Data th_0s: \n{Valid_Samples}")

## 모델 아키텍처 로드 및 세팅
Model = Func.AEModel()
Model.UseTrainData(TrainData)
Model.SetInputOutput(Input, Output)
Model.SetModel([5,4,4,5], activation='tanh')  # 노드 구조
Model.SetNormalization(mode='gaussian')
LossFunction = nn.MSELoss()
Optimizer = torch.optim.Adam(Model.parameters(), lr=LearnRate)
if UseL2Regularizer:
    Optimizer = torch.optim.Adam(Model.parameters(), lr=LearnRate, weight_decay=0.99)
# Optimizer=torch.optim.SGD(Model.parameters(),lr=LearnRate,momentum=0.99)
if UseLRScheduler:
    LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Optimizer, 'min', min_lr=1e-4,
                                                             factor=0.99,
                                                             threshold_mode='rel',
                                                             threshold=0.1,
                                                             patience=50)

Initial_Parameters = Model.state_dict()

# 입출력 나눔
TrainX, TrainY = TrainData[Input], TrainData[Output]
ValidX, ValidY = ValidData[Input], ValidData[Output]

# 정규화
TrainX, TrainY = Model.NormalizeInput(TrainX), Model.NormalizeOutput(TrainY)
ValidX, ValidY = Model.NormalizeInput(ValidX), Model.NormalizeOutput(ValidY)


# L, c값은 1개만 사용하므로 정규화 에러 수정해준다
# if 'L' in TrainX.columns:
#     TrainX['L'], ValidX['L'], = Ls[0], Ls[0]
#     TrainY['L'], ValidY['L'], = Ls[0], Ls[0]
# if 'c' in TrainX.columns:
#     TrainX['c'], ValidX['c'] = cs[0], cs[0]
#     TrainY['c'], ValidY['c'] = cs[0], cs[0]
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
Train_DLoader = DataLoader(TrainSet, batch_size=BatchSize, shuffle=DoShuffle)

# 모델 성능 기록표
Score_R2 = pd.DataFrame(np.zeros((Epochs, outD)), columns=Output, index=range(1, Epochs + 1))
Score_MSE = Score_R2
ModelScores = pd.concat([Score_R2, Score_MSE], axis=1, keys=['R2', 'MSE'])


## 학습 코드
Start = time.time()
for i in range(Epochs):
    for batchidx, batchdata in enumerate(Train_DLoader):
        trainX, trainY = batchdata  # 입출력 데이터 선언
        Pred = Model.forward(trainX)  # 출력
        Loss=LossFunction(Pred,trainY)
        Optimizer.zero_grad()  # Autograd 초기화
        Loss.backward()  # 역전파
        Optimizer.step()  # 가중치 수정
        if UseLRScheduler:
            # LRScheduler.step(Loss)  # 학습률 스케줄러
            LRScheduler.step(i+batchidx/len(Train_DLoader))
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{Epochs}, Loss={Loss.item():.6f}")
        if UseLRScheduler:
            print(f"Current learning rate={LRScheduler.state_dict()['_last_lr']}")

    # 매 Epoch마다 결정계수 검증
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
    #     Labels = pd.DataFrame(Labels.cpu().numpy(), columns=Output)
    #     Predictions = pd.DataFrame(Predictions.cpu().numpy(), columns=Output)
    #     for col in Output:
    #         ModelScores['R2', col].iloc[i] = R2(Labels[col], Predictions[col])
    #         ModelScores['MSE', col].iloc[i] = MSE(Labels[col], Predictions[col])

End = time.time()
Time = int(End - Start)
Final_Parameters = Model.state_dict()

with torch.no_grad():
    Model.eval()
    Labels = torch.zeros(ValidY.shape).cuda()
    Predictions = torch.zeros(ValidY.shape).cuda()

    for idx, ValidData in enumerate(ValidSet):
        validX, validY = ValidData
        Labels[idx, :] = validY
        PredY = Model.forward(validX)
        Predictions[idx, :] = PredY
        if (idx + 1) % 1000 == 0:
            print(f"Predicting values {idx + 1}/{len(ValidSet)}")

    Labels = pd.DataFrame(Labels.cpu().numpy(), columns=Output)
    Predictions = pd.DataFrame(Predictions.cpu().numpy(), columns=Output)

    # MSE와 R2는 정규화 상태로 계산한다
    for col in Output:
        ModelScores['R2', col].iloc[-1] = R2(Labels[col], Predictions[col])
        print(f"{col} R2: {ModelScores['R2', col].iloc[-1]:.6f}")
    for col in Output:
        ModelScores['MSE', col].iloc[-1] = MSE(Labels[col], Predictions[col])
        print(f"{col} MSE(Normalized): {ModelScores['MSE', col].iloc[-1]:.6f}")

    # 역정규화
    Labels = Model.InverseOutput(Labels)
    Predictions = Model.InverseOutput(Predictions)

    print(Labels)
    print(Predictions)
    with pd.option_context('display.max_columns', None):
        print(ModelScores['R2'])

print(f"Learning time {Time}sec.")
