import Functions as Func
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# input_size – The number of expected features in the input x
# hidden_size – The number of features in the hidden state h
# num_layers – Number of recurrent_type layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
# nonlinearity – The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
# fcbias – If False, then the layer does not use fcbias weights b_ih and b_hh. Default: True
# batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
# dropout – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
# bidirectional – If True, becomes a bidirectional RNN. Default: False


class RecurrentNet(torch.nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.num_features = kwargs['num_features']
		self.output_size = kwargs['output_size']
		self.hidden_size = kwargs['hidden_size']
		self.num_layers = kwargs['num_layers']
		self.nonlinearity = kwargs['nonlinearity']
		self.type = kwargs['type']
		self.num_fclayers = kwargs['num_fclayers'] if 'num_fclayers' in kwargs.keys() else 0
		self.dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 1
		self.bias = kwargs['fcbias'] if 'fcbias' in kwargs.keys() else True
		self.learningrate=kwargs['lr'] if 'lr' in kwargs.keys() else 1e-3
		self.optimizer=torch.optim.Adam(params=self.parameters(),lr=self.learningrate)
		
		if self.type == 'rnn' or 'RNN':
			self.rnn = torch.nn.RNN(input_size=self.num_features, hidden_size=self.hidden_size,
			                        num_layers=self.num_layers, bias=self.bias,
			                        batch_first=True, nonlinearity=self.nonlinearity
			                        , dropout=self.dropout)
		elif self.type == 'lstm' or 'LSTM':
			self.lstm = torch.nn.LSTM(input_size=self.num_features, hidden_size=self.hidden_size,
			                          num_layers=self.num_layers,
			                          bias=self.bias, batch_first=True, dropout=self.dropout)
		elif self.type=='gru' or 'GRU':
			self.gru = torch.nn.GRU(input_size=self.num_features, hidden_size=self.hidden_size,
			                          num_layers=self.num_layers,
			                          bias=self.bias, batch_first=True, dropout=self.dropout)
		
		self.FCLayers = torch.nn.ModuleList()
		if self.num_fclayers > 1:
			for i in range(self.num_fclayers):
				self.FCLayers.add_module(f"FC{i + 1}", torch.nn.Linear(self.hidden_size, self.hidden_size))
		
		self.FCOut = torch.nn.Linear(self.hidden_size, self.output_size, bias=True)
		self.Activation = torch.nn.ReLU()
	
	
	def forward(self, x):
		# (B,InSeq,Features)
		# InF=input_size로 설정함
		h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, requires_grad=True).to(x.device)
		c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, requires_grad=True).to(x.device)
		if self.type == 'rnn' or 'RNN':
			x, status = self.rnn(x, h0.detach())  # x는 모든 시점, status는 마지막 시점에 대한 은닉값
		elif self.type == 'lstm' or 'LSTM':
			x, status = self.lstm(x, (h0.detach(), c0.detach()))
		elif self.type=='gru' or 'GRU':
			x,status=self.gru(x,h0.detach())
		# x,status=self.lstm(x)
		# status # (Layers,B,H) 마지막 시점에서의 은닉값
		# (B,InSeq,NumHidden)
		# H=hidden_size로 설정함
		if self.num_fclayers > 1:
			for fclayer in self.FCLayers:
				x = fclayer(x)
				x = self.Activation(x)
		x = self.FCOut(x)
		# (B,InSeq,NumFeatures)
		return x


class Normalizer():
	def UseData(self, DF):
		self.DF = DF
		self.DFmax = DF.max()
		self.DFmin = DF.min()
		self.DFscale = DF.max() - DF.min()
	
	
	def Normalize(self, DF):
		return (DF - self.DFmin) / self.DFscale
	
	
	def Inverse(self, DF):
		return DF * self.DFscale + self.DFmin


def MyPlotTemplate():
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = 14
	plt.rcParams['mathtext.fontset'] = 'stix'


MBD = Func.MBD_Integrator()
Normalizer = Normalizer()
# Data = MBD.SC_Kin(tau=2.13, r=1.3, Lr=3)
# Data = Data.drop(columns=['Time', 'tau', 'r', 'L/r'])
Data = pd.read_csv("RecurDyn/Data.csv")
Data = Data.drop(columns=['No', 'TIME'])
print(Data)

Normalizer.UseData(Data)
Data = Normalizer.Normalize(Data)
Columns = Data.columns
Data = Data.to_numpy()
Data = torch.FloatTensor(Data)

# 시드 고정
torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True

InSeq = 10
OutSeq = 1
B = len(Data) - InSeq
Epoch = 200
NumFeatures = len(Columns)
BatchSize = 256
Shuffle = True
NumLayers = 4
NumFCLayers = 1
NumHidden = 256
Bias = True
LearningRate = 1e-3
Type = 'GRU'
Nonlinearity = 'relu'
Dropout = 0
TrainDataRate = 0.7

Net = RecurrentNet(num_features=NumFeatures, hidden_size=NumHidden, num_layers=NumLayers,
                   output_size=NumFeatures,
                   bias=Bias, nonlinearity=Nonlinearity, dropout=Dropout, type=Type, num_fclayers=NumFCLayers)
Optimizer = torch.optim.Adam(params=Net.parameters(), lr=LearningRate)
LossFunc = torch.nn.MSELoss()

X = torch.zeros((B, InSeq, NumFeatures))  # 3차원 텐서
Y = torch.zeros((B, OutSeq, NumFeatures))
for i in range(B):
	X[i, :, :] = Data[i:i + InSeq, :]
	Y[i, :, :] = Data[i + InSeq:i + InSeq + OutSeq, :]

Net = Net.cuda()  # GPU전송
X, Y = X.cuda(), Y.cuda()  # GPU전송
TrainIdx = int(X.shape[0] * TrainDataRate)  # 70% 학습시 사용
TrainSet = TensorDataset(X[:TrainIdx, :, :], Y[:TrainIdx, :, :])
TestSet = TensorDataset(X[TrainIdx:, :, :], Y[TrainIdx:, :, :])
TrainLoader = DataLoader(TrainSet, batch_size=BatchSize, shuffle=Shuffle)
TestLoader = DataLoader(TestSet, batch_size=BatchSize)

for i in range(Epoch):
	for batch in TrainLoader:
		Optimizer.zero_grad()
		trainx, trainy = batch
		label = trainy
		prediction = Net.forward(trainx)
		prediction = prediction[:, -1, :].view(trainy.shape[0], 1, NumFeatures)  # 마지막 시퀀스 시점에서의 예측 벡터
		Loss = LossFunc(label, prediction)
		Loss.backward()
		Optimizer.step()
	
	if (i + 1) % 1 == 0:
		print(f"Epoch={i + 1}, Loss={Loss.item():.6f}")
		print(label[0])
		print(prediction[0])
		print()

with torch.no_grad():
	Net = Net.cpu()
	Net.eval()
	Labels = Data
	
	# 자기회귀적 예측
	Predictions = torch.zeros((len(Data), NumFeatures))
	Predictions[:InSeq, :] = Labels[:InSeq, :]  # 첫 시퀀스에 대해서만 정보를 준다
	
	for i in range(B):
		input = Predictions[i:i + InSeq, :].view(1, InSeq, NumFeatures)
		prediction = Net.forward(input)  # 3차원 텐서 출력
		prediction = prediction[:, -1, :].view(OutSeq, NumFeatures)  # 마지막 시점 예측참조,
		Predictions[i + InSeq + OutSeq - 1, :] = prediction  # 인덱싱에 주의:
		print("Input Tensor:")
		print(input)
		print(f"\nTimestep {i + InSeq + 1} Label")
		print(Labels[i + InSeq + OutSeq - 1])  # 인덱싱에 주의:
		print(f"\nTimestep {i + +InSeq + 1} Prediction")
		print(prediction)
		print()
	
	# # 일반적 실수
	# Predictions = torch.zeros((len(Data), NumFeatures))
	# Predictions[:TrainIdx, :] = Labels[:TrainIdx, :]  # 훈련 시퀀스에 대해 정보를 준다
	# for i in range(len(Labels) - InSeq):
	# 	input = Predictions[i:i + InSeq, :].view(1, InSeq, NumFeatures)
	# 	prediction = Net.forward(input)  # 3차원 텐서 출력
	# 	prediction = prediction[:, -1, :].view(OutSeq, NumFeatures)  # 마지막 시점 예측참조,
	# 	Predictions[InSeq + OutSeq, :] = prediction
	
	Labels = pd.DataFrame(Labels.numpy(), columns=Columns)
	Predictions = pd.DataFrame(Predictions.numpy(), columns=Columns)

Labels = Normalizer.Inverse(Labels)
Predictions = Normalizer.Inverse(Predictions)

Labels.to_csv('Labels.csv')
Predictions.to_csv('Predictions.csv')

plt.figure(figsize=(19, 10))
MyPlotTemplate()
for i, feature in enumerate(Labels.columns):
	plt.subplot(3, 2, i + 1)
	plt.plot(Labels.index, Labels[feature], '-', c='k', label='Label', markersize=3)
	plt.plot(Labels.index, Predictions[feature], '-', c='r', label='Prediction', markersize=3)
	plt.vlines(TrainIdx, Labels[feature].min(), Labels[feature].max(), colors='b',
	           label='Train-Test Line', zorder=0, linewidth=2) #수직선
	plt.xlabel('Time steps'), plt.ylabel("Values")
	plt.title(feature)
	plt.grid()
	plt.legend(loc=1,fontsize=10)

plt.tight_layout()

ModelSummary = f"{Type},{InSeq}InSeq,{OutSeq}OutSeq,{Epoch}Epoch,{NumFeatures}Features,"
ModelSummary += f"{BatchSize}TrainingBatch,Shuffle{Shuffle},{NumLayers}RecurrentLayers,"
ModelSummary += f"{NumFCLayers}FCLayers,{NumHidden}CellHiddenSize,Bias{Bias},"
ModelSummary += f"{LearningRate}LearningRate,RecurrentNonlinear_{Nonlinearity} ,"
ModelSummary += f"{Dropout}DropoutRate,{TrainDataRate}TrainDataRate"

plt.savefig(f"{ModelSummary}.png")
plt.show()