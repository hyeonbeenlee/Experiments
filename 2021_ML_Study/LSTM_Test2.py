import pandas as pd
import pandas_datareader as pdr
import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os, sys

def Norm(DF,*args):
	if 'minmax' in args:
		return (DF-DF.min())/(DF.max()-DF.min())
	elif 'gaussian' in args:
		return (DF-DF.mean())/DF.std()

FuncModulePath = os.path.abspath('..')
FuncModulePath += "\DNN-MBD\Functions.py"
from shutil import copy2


copy2(FuncModulePath, os.path.abspath('.'))
import Functions as Func


Start = datetime.date(2000, 1, 1)
End = datetime.date.today()
RawData = pdr.DataReader('005930.KS', 'yahoo', Start, End)
RawData=RawData['Close']
# #주식 가격 변화량으로 셋팅
# for i in range(len(RawData)-1):
# 	RawData.iloc[i]=RawData.iloc[i+1]-RawData.iloc[i]
# RawData.drop(RawData.index[-1],inplace=True)
# 삼성전자 주가예측

Data = RawData.iloc[:4500]  # 종가만 사용한다
Data=(Data-RawData.min())/(RawData.max()-RawData.min()) #정규화


TestData=RawData.iloc[4500:]
TestData=(TestData-RawData.min())/(RawData.max()-RawData.min()) # 정규화

# DataNorm = (Data - Data.min()) / (Data.max() - Data.min())  # 정규화

class RNNNet(torch.nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		input_size = kwargs['input_size']
		output_size = kwargs['output_size']
		self.hidden_size = kwargs['hidden_size']
		self.num_layers = kwargs['num_layers']
		activation = kwargs['activation']
		dropout=kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
		bias = kwargs['fcbias'] if 'fcbias' in kwargs.keys() else True
		
		self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=self.hidden_size,
		                        num_layers=self.num_layers, bias=bias, batch_first=True, nonlinearity=activation
		                        ,dropout=dropout)
		self.lstm=torch.nn.LSTM(input_size=input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,
		                        bias=bias, batch_first=True,dropout=dropout)
		self.fc1 = torch.nn.Linear(self.hidden_size, output_size, bias=True)
	
	
	def forward(self, x):
		# (B,InL,InF)
		# InF=input_size로 설정함
		h0=torch.zeros(self.num_layers,x.shape[0],self.hidden_size,requires_grad=True).to(x.device)
		c0=torch.zeros(self.num_layers,x.shape[0],self.hidden_size,requires_grad=True).to(x.device)
		x, status = self.lstm(x,(h0.detach(),c0.detach())) #x는 모든 시점, status는 마지막 시점에 대한 은닉값
		# x,status=self.lstm(x)
		#status # (Layers,B,H) 마지막 시점에서의 은닉값
		# (B,InL,H)
		# H=hidden_size로 설정함
		x = self.fc1(x)  # (H->OutF)
		# (B,InL,OutL)
		# x=self.activation(x)
		# x=self.fc2(x)
		return x


Data = torch.FloatTensor(Data.to_numpy())
TestData=torch.FloatTensor(TestData.to_numpy())
PossibleTimestep = []
for i in range(len(Data)):
	if len(Data) % (i + 1) == 0:
		PossibleTimestep.append(i + 1)

print(PossibleTimestep)  # 9개 이용 1개 예측
print(Data)
InputUsage = 3;
OutputUsage = 1
InputData = [];
OutputData = []
Loops = int((len(Data) - InputUsage))  # 훈련데이터 배치 수
TestLoops=int((len(TestData) - InputUsage)) #시험데이터 배치 수
TrainX = torch.zeros((Loops, InputUsage, 1))
TrainY = torch.zeros((Loops, OutputUsage, 1))
TestX = torch.zeros((TestLoops, InputUsage, 1))
TestY = torch.zeros((TestLoops, OutputUsage, 1))

for i in range(Loops):
	SingleInputTensor = Data[i:i + InputUsage]  # 0~7 1D
	SingleOutputTensor = Data[i + InputUsage:i + InputUsage + OutputUsage]  # 8~10 1D
	TrainX[i, :, 0] = SingleInputTensor  # Batch,Timestep,FeatureIdx
	TrainY[i, :, 0] = SingleOutputTensor
	

for i in range(TestLoops):
	SingleInputTensor = TestData[i:i + InputUsage]  # 0~7 1D
	SingleOutputTensor = TestData[i + InputUsage:i + InputUsage + OutputUsage]  # 8~10 1D
	TestX[i, :, 0] = SingleInputTensor  # Batch,Timestep,FeatureIdx
	TestY[i, :, 0] = SingleOutputTensor

# https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
RNN = RNNNet(input_size=1, hidden_size=32, num_layers=3, bias=True,
             output_size=1, activation='tanh',dropout=0.1)
x = RNN.forward(TrainX)


Epoch = 20
Optimizer = torch.optim.Adam(params=RNN.parameters(), lr=1e-3)
LossFunc = torch.nn.MSELoss()

TrainX,TrainY=TrainX.cuda(),TrainY.cuda()
RNN=RNN.cuda()
Dataset=TensorDataset(TrainX,TrainY)
TrainDLoader=DataLoader(Dataset,batch_size=128,shuffle=False)

for i in range(Epoch):
	for Data in TrainDLoader:
		trainX,trainY=Data
		Optimizer.zero_grad()
		predY=RNN.forward(trainX)[:,-1,:] #마지막 타임스텝에서의 포워드 예측값만 따와서 펼침
		predY=predY.reshape(-1)
		trainY=trainY.view(-1)
		Loss = LossFunc(predY, trainY)
		Loss.backward()
		Optimizer.step()
	print(f"Epoch={i + 1}/{Epoch}, Loss={Loss.item():.8f}")

with torch.no_grad():
	RNN.eval()
	RNN = RNN.cpu()
	TrainX, TrainY = TrainX.cpu(), TrainY.cpu()
	# SingleInput=TrainX[-1,:,:] #마지막 배치의 단일 인풋
	# SingleOutput=RNN.forward(TrainX[-1,:,:].view(1,-1,1)) #Single batch, InSeqLen, Single feature
	# print(SingleInput) #9개 타임스텝으로 이루어진 입력값
	# SingleOutput=SingleOutput[:,-1,:].item() # 마지막 timestep에서의 1개 예측값
	# print(SingleOutput)  # 9개 타임스텝에서의 다음 예측들
	
	
	#전체 데이터 길이만큼의 영텐서 생성
	Predictions=torch.zeros(len(RawData))
	#정규화된 학습데이터입력을 넣어준다
	NormDF=(RawData[:Loops+InputUsage]-RawData.min())/(RawData.max()-RawData.min())
	Predictions[:Loops+InputUsage]=torch.FloatTensor(NormDF.to_numpy())
	
	# # 시험데이터에 대한 예측
	# for i in range(TestLoops):
	# 	SingleInput=Predictions[Loops+i:Loops+i+InputUsage].view(1,InputUsage,1)
	# 	pred=RNN.forward(SingleInput) # 각 스텝에 대한 예측 수행
	# 	pred=pred[:,-1,:].item() #마지막 스텝에 대한 예측rkqt
	# 	Predictions[Loops+i+InputUsage]=pred #만 기록
	#
	#
	# 	print(f"For time step {Loops+InputUsage+i}")
	# 	print(SingleInput)
	# 	print(f"Label: {((RawData - RawData.min()) / (RawData.max() - RawData.min())).iloc[Loops + i + InputUsage]}")
	# 	print(f"Prediction:{pred}\n")
		
	# 전체데이터에 대한 예측
	for i in range(Loops+TestLoops):
		SingleInput = Predictions[i:i + InputUsage].view(1, InputUsage, 1)
		pred = RNN.forward(SingleInput)  # 각 스텝에 대한 예측 수행
		Pred=pred
		pred = pred[:, -1, :].view(-1)[-1]  # 마지막 스텝에 대한 예측rkqt
		Predictions[i + InputUsage] = pred  # 만 기록
		
		
		print(f"For time step {InputUsage + i}")
		print(SingleInput)
		print(f"Prediction:\n{Pred}\n")
		print(
			f"Label: {((RawData - RawData.min()) / (RawData.max() - RawData.min())).iloc[i + InputUsage]}")
		
	
	
	
	
	
	
	
	
	
	# for i in range(TestLoops):
	
	
	
	
	
	
	
	
	
	
	Func.MyPlotTemplate()

	plt.plot(range(len(RawData)),Norm(RawData,'minmax'),'-',markersize=3,c='k')
	plt.plot(range(len(RawData)),Predictions,'-',c='r',markersize=3)
	plt.ylim(-0.1,1.1)
	## 테스트 데이터 인풋을 순수한 예측값으로 주는게 맞다...! Retry.

	plt.grid()
	plt.show()