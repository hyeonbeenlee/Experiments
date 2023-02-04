import torch
import torch.nn.functional as nnF
from torch.utils.data import TensorDataset, DataLoader, random_split
import Functions as Func
import time
import glob
from sklearn.metrics import r2_score as R2
import pandas as pd
import numpy as np


def DNN_Train_Save(Example, Case, **kwargs):
    np.random.seed(777)
    torch.manual_seed(777)

    if Example == 'Pendulum_Single':
        HiddenLs = 3
        Nodes = 256
        BatchSize = 256
        Epochs = 100
        LearnRate = 1e-3
        DoShuffle = True
        Input = ['Time', 'L', 'c', 'th_0']
        Output = ['th', 'dth', 'ddth']

    elif Example == 'Pendulum_Double':
        HiddenLs = 4
        Nodes = 64
        BatchSize = 1024
        Epochs = 400
        LearnRate = 1e-4
        DoShuffle = True
        Input = ['Time', 'L1', 'L2', 'dth1_0', 'dth2_0']
        Output = ['th1', 'th2', 'dth1', 'dth2']

    elif Example == 'SliderCrankKin':
        HiddenLs = 2
        Nodes = 128
        BatchSize = 64
        Epochs = 200
        LearnRate = 1e-3
        DoShuffle = True
        Input = ['Time', 'tau', 'r', 'L/r']
        Output = ['th', 'pi', 'dpi', 'ddpi', 'xb', 'dxb', 'ddxb']

    inD, outD = len(Input), len(Output)

    # 데이터프레임으로 병합

    if ('generate_data' in kwargs.keys()) and (kwargs['generate_data'][0]):
        K=kwargs['generate_data'][1]
        MBD = Func.MBD_Integrator()
        DataFrames = []
        Count = 0
        Ls = np.linspace(0.1, 0.2, K, endpoint=True)
        cs = np.linspace(0, 0.15, K, endpoint=True)
        th_0s = np.linspace(-np.pi / 2, np.pi / 2, K, endpoint=True)
        for L in Ls:
            for c in cs:
                for th_0 in th_0s:
                    DataFrames.append(MBD.Pendulum_Single(L=L, c=c, th_0=th_0))
                    Count += 1
                    if Count % 100 == 0:
                        print(f"Generating Train Data {Count}")
        TrainData = pd.concat(DataFrames, keys=range(len(DataFrames)))

    else:
        TrainData = Func.CollectConcat(f"MBD Data/{Example}/{Case}")

    #랜덤 데이터 추가
    if 'ADDRNDM' in Case:
        TrainData=pd.concat([TrainData,Func.CollectConcat('MBD Data/Pendulum_Single/2K_AdditionalRandom')],ignore_index=True)
    print(TrainData)
    ValidData = Func.CollectConcat(f"MBD Data/{Example}/TestData")

    Model = Func.DNNModel()
    Model.UseTrainData(TrainData)
    Model.SetInputOutput(Input, Output)
    Model.SetModel(input_dim=inD, nodes=Nodes, output_dim=outD, hidden_layers=HiddenLs, activation='relu')
    Optimizer = torch.optim.Adam(Model.parameters(), lr=LearnRate)

    # 입출력 나눔
    TrainX, TrainY = TrainData[Input], TrainData[Output]
    ValidX, ValidY = ValidData[Input], ValidData[Output]

    # 정규화
    TrainX, TrainY = Model.NormalizeInput(TrainX,mode='gaussian'), Model.NormalizeOutput(TrainY,mode='gaussian')
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
    Train_DLoader = DataLoader(TrainSet, batch_size=BatchSize, shuffle=DoShuffle)

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
    End = time.time()
    Time = int(End - Start)

    ## 최종 결정계수 평가
    if ('end_evaluate' in kwargs.keys()) and (kwargs['end_evaluate']):
        with torch.no_grad():
            Model.eval()
            Labels = torch.zeros(TrainY.shape).cuda()
            Predictions = torch.zeros(TrainY.shape).cuda()

            for idx, ValidData in enumerate(ValidSet):
                validX, validY = ValidData
                Labels[idx, :] = validY
                PredY = Model(validX)
                Predictions[idx, :] = PredY
                print(f"Predicting values {idx + 1}/{len(ValidSet)}")

            Labels = pd.DataFrame(Labels.cpu(), columns=Output)
            Predictions = pd.DataFrame(Predictions.cpu(), columns=Output)
            for col in Output:
                print(f"{col} R2: {R2(Labels[col], Predictions[col]):.6f}")



    print(f"Learning time {Time}sec.")

    if 'save' in kwargs.keys() and kwargs['save']:
        torch.save(Model.state_dict(), f"DNN Model/{Example}/{Case}.pt")  # 모델 파라미터 저장
        print(f'Example-{Example}, Case-{Case} Model Saved.')


if __name__=='__main__':
    DNN_Train_Save('Pendulum_Single','3K',generate_data=(True,3),end_evaluate=True,save=False)