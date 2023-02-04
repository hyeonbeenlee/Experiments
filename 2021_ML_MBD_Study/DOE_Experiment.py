import matplotlib.pyplot as plt
import torch
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
    global HiddenLs, Nodes, BatchSize, Epochs, LearnRate, DoShuffle, Input, Output, inD, outD
    # np.random.seed(777)
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)
    HiddenLs = 3
    Nodes = 256
    BatchSize = 256 # 256
    Epochs = 300
    LearnRate = 1e-3
    DoShuffle = True
    Input = ['Time', 'th_0']
    Output = ['th']
    # Input = ['Time', 'L', 'c', 'th_0']
    # Output = ['th', 'dth', 'ddth']
    inD, outD = len(Input), len(Output)


SetHyperParam()

## MBD 학습데이터 생성 및 병합
MBD = Func.MBD_Integrator()
DataFrames = []
Count = 0
Ls = [0.13]  # np.linspace(0.1, 0.2, K, endpoint=True)
cs = [0.1]  # np.linspace(0, 0.15, K, endpoint=True)
K = 11
th_0s = np.linspace(-np.pi / 2, np.pi / 2, K, endpoint=True)
# th_0s = np.sort(np.random.random(K)) * np.pi - np.pi / 2
Train_Samples = th_0s
for L in Ls:
    for c in cs:
        for th_0 in th_0s:
            DataFrames.append(MBD.Pendulum_Single(L=L, c=c, th_0=th_0))
            Count += 1
            print(f"Generating MBD Train Data {Count}")
TrainData = pd.concat(DataFrames, keys=range(1, len(DataFrames) + 1))
TrainData_Original = TrainData

## MBD 검증데이터 생성 및 병합
MBD = Func.MBD_Integrator()
DataFrames = []
Count = 0
K = 21
th_0s = np.sort(np.random.random(K)* np.pi - np.pi/2) #랜덤
# th_0s=np.linspace(-np.pi/2,np.pi/2,K,endpoint=True) #균일
Valid_Samples = th_0s
for L in Ls:
    for c in cs:
        for th_0 in th_0s:
            DataFrames.append(MBD.Pendulum_Single(L=L, c=c, th_0=th_0))
            Count += 1
            print(f"Generating MBD Train Data {Count}")
ValidData = pd.concat(DataFrames, keys=range(1, len(DataFrames) + 1))
ValidData_Original = ValidData

## 생성된 학습 및 검증 데이터 출력, 중복 검사
print(f"Train Data\n{TrainData}\n")
print(f"Valid Data\n{ValidData}\n")
Warning = False
for datapoint in Valid_Samples:
    if datapoint in Train_Samples:
        Warning = True
if Warning:
    print('학습데이터와 검증데이터가 분리되지 않음!')
else:
    print('학습데이터와 검증데이터가 분리됨')
print(f"Train Data th_0s: \n{Train_Samples}")
print(f"Valid Data th_0s: \n{Valid_Samples}")

## 모델 아키텍처 로드 및 세팅
Model = Func.DNNModel()
Model.UseTrainData(TrainData)
Model.SetInputOutput(Input, Output)
Model.SetModel(input_dim=inD, nodes=Nodes, output_dim=outD, hidden_layers=HiddenLs, activation='relu')
Optimizer = torch.optim.Adam(Model.parameters(), lr=LearnRate)
Initial_Parameters = Model.state_dict()

# 입출력 나눔
TrainX, TrainY = TrainData[Input], TrainData[Output]
ValidX, ValidY = ValidData[Input], ValidData[Output]

# 정규화
TrainX, TrainY = Model.NormalizeInput(TrainX, mode='minmax'), Model.NormalizeOutput(TrainY, mode='minmax')
ValidX, ValidY = Model.NormalizeInput(ValidX, mode='minmax'), Model.NormalizeOutput(ValidY, mode='minmax')

# L, c값은 1개만 사용하므로 정규화 에러 수정해준다
if 'L' in TrainX.columns:
    TrainX['L'], ValidX['L'], = Ls[0], Ls[0]
if 'c' in TrainX.columns:
    TrainX['c'], ValidX['c'] = cs[0], cs[0]

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

Score_R2 = pd.DataFrame(np.zeros((Epochs, outD)), columns=Output, index=range(1, Epochs + 1))
Score_MSE = Score_R2
ModelScores = pd.concat([Score_R2, Score_MSE], axis=1, keys=['R2', 'MSE'])

## 학습 코드
Start = time.time()
for i in range(Epochs):
    for batchidx, batchdata in enumerate(Train_DLoader):
        trainX, trainY = batchdata  # 입출력 데이터 선언
        Pred = Model.forward(trainX)  # 출력
        Loss = nnF.mse_loss(Pred, trainY)  # 손실계산
        Optimizer.zero_grad()  # Autograd 초기화
        Loss.backward()  # 역전파
        Optimizer.step()  # 가중치 수정

    print(f"{i + 1}/{Epochs}, Loss={Loss.item():.6f}")

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
        ModelScores['MSE', col].iloc[-1] = MSE(Labels[col], Predictions[col])
        print(f"{col} R2: {ModelScores['R2', col].iloc[-1]:.6f}")
        print(f"{col} MSE: {ModelScores['MSE', col].iloc[-1]:.6f}")

    Labels = Model.InverseOutput(Labels, mode='minmax')
    Predictions = Model.InverseOutput(Predictions, mode='minmax')

print(ModelScores)
print(f"Learning time {Time}sec.")

##플로팅 코드
Func.MyPlotTemplate()

Titles = [r'$R^2_\theta$', r'$R^2_\dot\theta$', r'$R^2_\ddot\theta$']
for idx, colname in enumerate(Output):
    plt.subplot(2, 3, idx + 1)
    plt.scatter(Labels[colname], Predictions[colname], c='k', s=3)
    plt.xlabel('Labels')
    plt.ylabel('Predictions')
    plt.title(Titles[idx])
    plt.grid()

plt.subplot(2, 3, 4)
plt.plot(range(Epochs), ModelScores['R2', 'th'], '-o', label=r'$R^2$', c='r', ms=3)
plt.xlabel('Epochs')
plt.ylabel(r'$R^2$ Value')
plt.legend()
plt.grid()

plt.subplot(2, 3, 5)
plt.plot(range(Epochs), ModelScores['MSE', 'th'], '-o', label=r'$MSE$', c='b', ms=3)
plt.xlabel('Epochs')
plt.ylabel('MSE Error(Inversed)')
plt.legend()
plt.grid()

plt.subplot(2, 3, 6)
Index = 3  # Valid Data의 key 숫자-1 (인덱스)를 넣는다
plt.title(r"$\theta_0=$" + f"{ValidData_Original['th_0'].loc[Index].iloc[0]}")
plt.xlabel('Time (sec)')
plt.ylabel(r'$\theta (rad)$')
plt.plot(np.linspace(0, 2, 201, endpoint=True), Labels.iloc[201 * (Index - 1):201 * Index, 0])
plt.plot(np.linspace(0, 2, 201, endpoint=True), Predictions.iloc[201 * (Index - 1):201 * Index, 0])
plt.grid()

plt.tight_layout()

## 3D 검증데이터 표면 시각화
Time = np.linspace(0, 2, 201, endpoint=True)  # 시간배열
TH_0 = Valid_Samples  # 샘플링된 검증 데이터 theta_0 배열
Pred_THSurf = np.zeros((len(Time), len(TH_0)))  # (시간축, theta_0축)
Label_THSurf = np.zeros((len(Time), len(TH_0)))
Err_THSurf = np.zeros((len(Time), len(TH_0)))
for th_0idx, timeidx in enumerate(range(0, len(Predictions), 201)):
    Label_THSurf[:, th_0idx] = Labels.iloc[timeidx:timeidx + 201, 0]
    Pred_THSurf[:, th_0idx] = Predictions.iloc[timeidx:timeidx + 201, 0]
    Error=(Labels.iloc[timeidx:timeidx + 201, 0]-Predictions.iloc[timeidx:timeidx + 201, 0]).abs()
    Err_THSurf[:, th_0idx] = Error
TH_0, Time = np.meshgrid(TH_0, Time)

fig2 = plt.figure(figsize=(19.2, 10.8))
ax1 = fig2.add_subplot(131, projection='3d')
ax2 = fig2.add_subplot(132, projection='3d')
ax3 = fig2.add_subplot(133,projection='3d')

ax1.plot_surface(Time, TH_0, Pred_THSurf, color='b')
ax1.plot_surface(Time, TH_0, Label_THSurf, color='r')
import matplotlib as mpl
fakeline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
fakeline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
ax1.legend([fakeline1, fakeline2], ['Prediction Surface', 'Label Surface'])
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel(r'$\theta_0 (rad)$')
ax1.set_zlabel(r'$\theta (rad)$')
ax1.set_xlim([0,2])
ax1.set_ylim([-np.pi/2,np.pi/2])
title1 = f"Validation Dataset Surface\n"
title1 += r"$R^2_\theta$="
title1 += f"{ModelScores['R2', 'th'].iloc[-1]:.6f}, MSE(Normalized)={ModelScores['MSE', 'th'].iloc[-1]:.6f}"
ax1.set_title(title1)

for i in range(int(len(TrainData_Original) / len(Time))):  # range(K)
    ax2.plot(TrainData_Original['Time'].loc[i + 1], TrainData_Original['th_0'].loc[i + 1],
             TrainData_Original['th'].loc[i + 1], color='k')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel(r'$\theta_0 (rad)$')
ax2.set_zlabel(r'$\theta (rad)$')
ax3.set_xlim([0,2])
ax2.set_ylim((-np.pi/2,np.pi/2))
title2 = f"Used {len(Train_Samples)} Train Datasets\n"
title2 += r"$\theta_0(rad)=[$"
for i in range(len(Train_Samples)):
    if (i + 1) < len(Train_Samples):
        title2 += f"{Train_Samples[i]:.4f}, "
    if (i + 1) % 5 == 0:
        title2 += "\n"
    if (i + 1) == len(Train_Samples):
        title2 += f"{Train_Samples[i]:.4f}"
title2 += r"$]$"
ax2.set_title(title2)
fig2.tight_layout()


ax3.plot_surface(Time,TH_0,Err_THSurf,cmap='gist_rainbow')
ax3.set_xlabel('Time (sec)')
ax3.set_ylabel(r'$\theta_0 (rad)$')
ax3.set_zlabel('Absolute Error(Inverse Normalized)')
ax3.set_xlim([0,2])
ax3.set_ylim([-np.pi/2,np.pi/2])
title3 = f"Error Contour\n"
title3 += f"MSE(Normalized)={ModelScores['MSE', 'th'].iloc[-1]:.6f}"
ax3.set_title(title3)

plt.show()
