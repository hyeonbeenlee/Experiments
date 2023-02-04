import pandas as pd
import pandas_datareader as pdr
import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os, sys


FuncModulePath = os.path.abspath('..')
FuncModulePath += "\DNN-MBD\Functions.py"
from shutil import copy2


copy2(FuncModulePath, os.path.abspath('.'))
import Functions as Func




Start = datetime.date(2000, 1, 1)
End = datetime.date.today()
Data = pdr.DataReader('005930.KS', 'yahoo', Start, End)
DataNorm = (Data - Data.min()) / (Data.max() - Data.min())  # 정규화
print(Data)

# plt.plot(Data.index,Data['Close'])
# plt.grid()
# plt.tight_layout()
# plt.show()
Train, Valid = DataNorm[:4500], DataNorm[4500:]
Train = torch.FloatTensor(Train['Close'].to_numpy())
Train = torch.reshape(Train, (450, 10, 1))  # batch, timestep, feature

print(Train)

input_size = 1
hidden_size = 16
cell = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=3, batch_first=True, bias=True)
outputs, status = cell(Train)
print(outputs.shape)  # 모든 타임스텝에서의 은닉층값
print(status.shape)  # 마지막 타임스텝에서의 은닉층값


class RNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers):
		super().__init__()
		self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
		                        num_layers=num_layers, bias=True, batch_first=True,nonlinearity='relu')
		self.fc = torch.nn.Linear(hidden_size, output_size, bias=True)
	
	
	def forward(self, x):
		x, status = self.rnn(x)
		x = self.fc(x)
		return x


net = RNN(input_size, hidden_size, 1, 5).cuda()
Train=Train.cuda()
lossfunc=torch.nn.MSELoss()
optim=torch.optim.Adam(net.parameters(),lr=1e-3)
outputs=net.forward(Train)


epoch=3
for i in range(epoch):
	optim.zero_grad()
	output=net.forward(Train)
	print(output.shape)
	loss=lossfunc(output.view(-1),Train.view(-1))
	loss.backward()
	optim.step()
	print(f"Epoch {i+1}/{epoch}, Loss={loss.item():.5f}")

with torch.no_grad():
	TrainforPlot=Train.view(-1).cpu().numpy()
	Prediction=net.forward(Train).view(-1).cpu().numpy()
	TrainforPlot=pd.DataFrame(TrainforPlot)
	Prediction=pd.DataFrame(Prediction)
	TrainforPlot=TrainforPlot*(Data['Close'].max()-Data['Close'].min())+Data['Close'].min()
	Prediction=Prediction*(Data['Close'].max()-Data['Close'].min())+Data['Close'].min()
	PreInput=(Data['Close'].iloc[4500:5455]-Data['Close'].min())/(Data['Close'].max()-Data['Close'].min())
	PreInput=torch.FloatTensor(PreInput).view(-1,5,1).cuda()
	PreOutput=net.forward(PreInput)
	PreOutput=PreOutput.cpu().view(-1).numpy()
	PreOutput=pd.DataFrame(PreOutput)*(Data['Close'].max()-Data['Close'].min())+Data['Close'].min()

# print(TrainforPlot)
# print(Prediction)

plt.plot(Data.index, Data['Close'])
plt.plot(Data.index[:4500], Prediction[0])
plt.plot(Data.index[4500:5455],PreOutput[0])
plt.grid()
plt.show()
