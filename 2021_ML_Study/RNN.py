import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split as RandomSplit
import matplotlib.pyplot as plt


class MyRNNClass(torch.nn.Module):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        super().__init__()

    def forward(self, X):
        # (B,InSeq,Features)
        # InF=input_size로 설정함
        h0=torch.zeros((self.recurrent_layers,X.shape[0],self.NumInfeatures),requires_grad=True).to(device)
        c0=torch.zeros((self.recurrent_layers,X.shape[0],self.NumInfeatures),requires_grad=True).to(device)
        if self.recurrent_type=='rnn':
            X,status=self.RecurrentLayer(X,h0.detach())
        elif self.recurrent_type=='lstm':
            X,status=self.RecurrentLayer(X,(h0.detach(),c0.detach()))
        elif self.recurrent_type=='gru':
            X,status=self.RecurrentLayer(X,h0.detach())
        # x,status=self.lstm(x)
        # status # (Layers,B,H) 마지막 시점에서의 은닉값
        # (B,InSeq,NumHidden)
        # H=hidden_size로 설정함
        X = self.InputLayer(X)
        X = self.fcActivation(X)
        for hiddenlayer in self.FCHiddenLayersModule:
            X = hiddenlayer(X)
            X = self.fcActivation(X)
        X = self.OutputLayer(X)
        # (B,InSeq,NumFeatures)
        return X

    def MyPlotTemplate(self):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        plt.rcParams['mathtext.fontset'] = 'stix'

    def FixSeed(self, seednum):
        self.seednum = seednum
        np.random.seed(seednum)
        torch.manual_seed(seednum)
        torch.cuda.manual_seed(seednum)
        torch.Generator().manual_seed(seednum)  # Data split
        torch.backends.cudnn.deterministic = True  # BackPropagation
        print(f"Fixed all stochastic seeds to {seednum}.")

    def SetHyperParams(self, **kwargs):
        # Define Hyperparams
        self.Numepochs = kwargs['epochs']
        self.Numfcnodes = kwargs['fcnodes']
        self.Numfchiddenlayers = kwargs['fchiddenlayers']
        self.Numbatchsize = kwargs['batchsize']
        self.learningrate = kwargs['lr']
        self.fcbias = kwargs['fcbias']  # Boolean
        # Nonlinear activation
        self.fcactivationtype = kwargs['fcactivation'].lower()
        if self.fcactivationtype == 'relu':
            self.fcActivation = torch.nn.ReLU()
        if self.fcactivationtype == 'tanh':
            self.fcActivation = torch.nn.Tanh()
        if self.fcactivationtype == 'sigmoid':
            self.fcActivation = torch.nn.Sigmoid()

        # Loss
        self.lossfunctype = kwargs['loss'].lower()
        if self.lossfunctype == 'mse':
            self.LossFunction = torch.nn.MSELoss()
        if self.lossfunctype == 'l1':
            self.LossFunction = torch.nn.L1loss()

        # Define Layers at DEVICE
        self.InputLayer = torch.nn.Linear(self.NumInfeatures, self.Numfcnodes, bias=self.fcbias).to(self.device)
        self.FCHiddenLayersModule = torch.nn.ModuleList()
        for i in range(self.Numfchiddenlayers):
            self.FCHiddenLayersModule.add_module(f"HiddenLayer{i + 1:d}",
                                                 torch.nn.Linear(self.Numfcnodes, self.Numfcnodes, bias=self.fcbias).to(
                                                     self.device))
        self.OutputLayer = torch.nn.Linear(self.Numfcnodes, self.NumOutfeatures).to(self.device)

        # Optimizer
        self.optimizertype = kwargs['optimizer'].lower()
        if self.optimizertype == 'adam':
            self.Optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learningrate)
            # weight_decay(regularization)
        if self.optimizertype == 'sgd':
            self.Optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learningrate)
            # momentum

        # Recurrent Layers
        self.recurrent_hiddensize = kwargs['recurrent_hiddensize']
        self.recurrent_layers = kwargs['recurrent_layers']
        self.recurrent_dropout = kwargs['recurrent_dropout']
        self.recurrent_bias=kwargs['recurrent_bias']
        # Nonlinearity
        self.recurrent_nonlinearity = kwargs['recurrent_nonlinearity'].lower()
        if self.recurrent_nonlinearity=='relu':
            self.RecurrentNonlinearity=torch.nn.ReLU()
        elif self.recurrent_nonlinearity=='tanh':
            self.RecurrentNonlinearity=torch.nn.Tanh()
        # Recurrent type
        self.recurrent_type = kwargs['recurrent_type'].lower()
        if self.recurrent_type == 'rnn':
            self.RecurrentLayer = torch.nn.RNN(input_size=self.NumInfeatures, hidden_size=self.recurrent_hiddensize,
                                               num_layers=self.recurrent_layers, bias=self.recurrent_bias,
                                               batch_first=True, nonlinearity=self.recurrent_nonlinearity
                                               , dropout=self.recurrent_dropout)
        elif self.recurrent_type == 'lstm':
            self.RecurrentLayer = torch.nn.LSTM(input_size=self.NumInfeatures, hidden_size=self.recurrent_hiddensize,
                                               num_layers=self.recurrent_layers, bias=self.recurrent_bias,
                                               batch_first=True, dropout=self.recurrent_dropout)
        elif self.recurrent_type== 'gru':
            self.RecurrentLayer=torch.nn.GRU(input_size=self.NumInfeatures, hidden_size=self.recurrent_hiddensize,
                                               num_layers=self.recurrent_layers, bias=self.recurrent_bias,
                                               batch_first=True, dropout=self.recurrent_dropout)


        print("Hyperparameters and layer modules are set.")

    def UseDataframe(self, pdDF, **kwargs):
        # Save Raw
        self.RawDF = pdDF
        self.DFlength = pdDF.shape[0]
        self.InputColumns = kwargs['inputcols']
        self.OutputColumns = kwargs['outputcols']
        self.NumInfeatures = len(self.InputColumns)
        self.NumOutfeatures = len(self.OutputColumns)

        self.DFmax = pdDF.max()
        self.DFmin = pdDF.min()
        self.DFstd = pdDF.std()
        self.DFmean = pdDF.mean()
        print(
            f"Loaded dataframe info: Length {self.DFlength} with {self.NumInfeatures} In-features & {self.NumOutfeatures} Out-features.")

    def ProcessData(self, **kwargs):

        # Configure split settings
        self.shuffle = kwargs['shuffle']
        self.traindatarate = kwargs['traindatarate']
        self.validdatarate = kwargs['validdatarate'] if 'validdatarate' in kwargs.keys() else 0
        self.testdatarate = 1 - self.traindatarate if 'validdatarate' not in kwargs.keys() else 1 - (
                self.traindatarate + self.validdatarate)
        self.normalizetype = kwargs['normalizetype']

        self.trainlength = int(self.RawDF.shape[0] * self.traindatarate)
        self.validlength = int(self.RawDF.shape[0] * self.validdatarate) if 'validdatarate' in kwargs.keys() else 0
        self.testlength = self.RawDF.shape[0] - self.trainlength if 'validdatarate' not in kwargs.keys() else \
        self.RawDF.shape[0] - (
                self.trainlength + self.validlength)

        # Normalize and split
        if self.normalizetype == 'minmax':
            self.NormalizedDF = (self.RawDF - self.DFmin) / (self.DFmax - self.DFmin)
        elif self.normalizetype == 'gaussian':
            self.NormalizedDF = (self.RawDF - self.DFmean) / self.DFstd
        self.NormalizedDF = self.NormalizedDF.fillna(0)  # Fill 0

        self.NormalizedTx = torch.FloatTensor(self.NormalizedDF[self.InputColumns].to_numpy()).to(self.device)
        self.NormalizedTy = torch.FloatTensor(self.NormalizedDF[self.OutputColumns].to_numpy()).to(self.device)
        self.NormalizedTDS = TensorDataset(self.NormalizedTx, self.NormalizedTy)
        self.NormalizedTDS_Train, self.NormalizedTDS_Valid, self.NormalizedTDS_Test = RandomSplit(self.NormalizedTDS,
                                                                                                  [self.trainlength,
                                                                                                   self.validlength,
                                                                                                   self.testlength])
        self.NormalizedTDL_Train = DataLoader(self.NormalizedTDS_Train, batch_size=self.Numbatchsize,
                                              shuffle=self.shuffle)
        if self.validdatarate > 0:
            self.NormalizedTDL_Valid = DataLoader(self.NormalizedTDS_Valid, batch_size=len(self.NormalizedTDS_Valid))
        self.NormalizedTDL_Test = DataLoader(self.NormalizedTDS_Test, batch_size=len(self.NormalizedTDS_Test))
        print("Data preprocessing done.")

    def Train(self, **kwargs):
        if self.seednum:
            self.FixSeed(self.seednum)
        print("Starting model training\n")
        StartTime = time.time()
        self.SetMode('train')

        if self.validdatarate > 0:
            validperiod = kwargs['validperiod']

        printrate = kwargs['printrate']

        self.TrainLossArray = np.zeros(self.Numepochs)  # GPU메모리 문제로 np array 사용
        self.ValidLossArray = np.zeros(self.Numepochs)

        for i in range(self.Numepochs):
            for batch in self.NormalizedTDL_Train:
                self.Optimizer.zero_grad()
                trainx, trainy = batch
                label = trainy
                prediction = self.forward(trainx)
                Loss = self.LossFunction(label, prediction)
                Loss.backward()
                self.Optimizer.step()

                # Record
                self.TrainLossArray[i] = Loss

                # Delete from gpu memory
                del batch, trainx, trainy, label, prediction

            if (i + 1) % printrate == 0:
                print(f"Epoch {i + 1}, Train Loss={Loss:.6f}")

            if (self.validdatarate > 0) and ('validperiod' in kwargs.keys()) and (validperiod != 0):
                if (i + 1) % validperiod == 0:
                    for batch in self.NormalizedTDL_Valid:
                        validx, validy = batch
                        vlabel = validy
                        vprediction = self.forward(validx)
                        vLoss = self.LossFunction(vlabel, vprediction)

                        # Record
                        self.ValidLossArray[i] = vLoss

                        if (i + 1) % printrate == 0:
                            print(f"Validation Loss={vLoss:.6f}\n")

                        # Delete from gpu memory
                        del batch, validx, validy, vprediction, vLoss

        EndTime = time.time()
        TrainTime = EndTime - StartTime
        print(f"Model training finished in {TrainTime:.1f}sec.")

    def SaveModel(self, path):
        Dict = {
            # Hyperparms
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.Optimizer.state_dict(),
            'inputcols': self.InputColumns,
            'outputcols': self.OutputColumns,
            'epochs': self.Numepochs,
            'nodes': self.Numfcnodes,
            'hiddenlayers': self.Numfchiddenlayers,
            'batchsize': self.Numbatchsize,
            'learningrate': self.learningrate,
            'fcbias': self.fcbias,
            'activation': self.fcactivationtype,
            'normalizetype': self.normalizetype,
            'traindatarate': self.traindatarate,
            'validdatarate': self.validdatarate,
            'testdatarate': self.testdatarate,
            'shuffle': self.shuffle,
            'loss': self.lossfunctype,
            'optimizer': self.optimizertype,

            # Loss log
            'TrainLossArray': self.TrainLossArray,
            'ValidLossArray': self.ValidLossArray,

            # Data info
            'DFmax': self.DFmax,
            'DFmin': self.DFmin,
            'DFmean': self.DFmean,
            'DFstd': self.DFstd
        }
        torch.save(Dict, path)
        print(f"Model saved: {path}\n")

    def LoadModel(self, path, **kwargs):
        SavePoint = torch.load(path)

        # Hyperparams
        self.InputColumns = SavePoint['inputcols']
        self.NumInfeatures = len(self.InputColumns)
        self.OutputColumns = SavePoint['outputcols']
        self.NumOutfeatures = len(self.OutputColumns)
        self.Numepochs = SavePoint['epochs']
        self.Numfcnodes = SavePoint['nodes']
        self.Numfchiddenlayers = SavePoint['hiddenlayers']
        self.Numbatchsize = SavePoint['batchsize']
        self.learningrate = SavePoint['learningrate']
        self.fcbias = SavePoint['fcbias']
        self.fcactivationtype = SavePoint['activation']
        self.normalizetype = SavePoint['normalizetype']
        self.traindatarate = SavePoint['traindatarate']
        self.validdatarate = SavePoint['validdatarate']
        self.shuffle = SavePoint['shuffle']
        self.lossfunctype = SavePoint['loss']
        self.TrainLossArray = SavePoint['TrainLossArray']
        self.ValidLossArray = SavePoint['ValidLossArray']
        self.optimizertype = SavePoint['optimizer']

        # Data Info
        self.DFmax = SavePoint['DFmax']
        self.DFmin = SavePoint['DFmin']
        self.DFmean = SavePoint['DFmean']
        self.DFstd = SavePoint['DFstd']

        # Reproduce Nonlinear activation
        if self.fcactivationtype == 'relu':
            self.fcActivation = torch.nn.ReLU()
        if self.fcactivationtype == 'tanh':
            self.fcActivation = torch.nn.Tanh()
        if self.fcactivationtype == 'sigmoid':
            self.fcActivation = torch.nn.Sigmoid()

        # Reproduce Loss
        if self.lossfunctype == 'mse':
            self.LossFunction = torch.nn.MSELoss()
        if self.lossfunctype == 'l1':
            self.LossFunction = torch.nn.L1loss()

        # Reproduce Layers at DEVICE
        self.InputLayer = torch.nn.Linear(self.NumInfeatures, self.Numfcnodes, bias=self.fcbias).to(self.device)
        self.FCHiddenLayersModule = torch.nn.ModuleList()
        for i in range(self.Numfchiddenlayers):
            self.FCHiddenLayersModule.add_module(f"HiddenLayer{i + 1:d}",
                                                 torch.nn.Linear(self.Numfcnodes, self.Numfcnodes, bias=self.fcbias).to(
                                                     self.device))
        self.OutputLayer = torch.nn.Linear(self.Numfcnodes, self.NumOutfeatures).to(self.device)

        # Reproduce Optimizer
        if self.optimizertype == 'adam':
            self.Optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learningrate)
            # weight_decay(regularization)
        if self.optimizertype == 'sgd':
            self.Optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learningrate)
            # momentum

        self.load_state_dict(SavePoint['model_state_dict'])
        self.Optimizer.load_state_dict(SavePoint['optimizer_state_dict'])

        print(f"Model loaded: {path}")

        # Reproduce train/valid/test dataloader
        if 'rawdata' in kwargs.keys():
            self.RawDF = kwargs['rawdata']
            self.ProcessData(normalizetype=self.normalizetype, traindatarate=self.traindatarate,
                             validdatarate=self.validdatarate, shuffle=self.shuffle)

    def SetMode(self, args):
        if args[0].lower() == 'train':
            self.train()
        elif args[1].lower() == 'eval':
            self.eval()

    def EvaluateTestData(self, **kwargs):
        self.SetMode('eval')

        with torch.no_grad():
            for batch in self.NormalizedTDL_Test:
                testx, testy = batch
                testprediction = self.forward(testx)

            if kwargs['inverse']:
                testprediction = np.array(testprediction.to('cpu'))
                testprediction = pd.DataFrame(testprediction, columns=self.OutputColumns)
                testprediction = testprediction * (self.DFmax[self.OutputColumns] - self.DFmin[self.OutputColumns]) + \
                                 self.DFmin[self.OutputColumns]

                testy = np.array(testy.to('cpu'))
                testy = pd.DataFrame(testy, columns=self.OutputColumns)
                testy = testy * (self.DFmax[self.OutputColumns] - self.DFmin[self.OutputColumns]) + \
                        self.DFmin[self.OutputColumns]

            else:
                testprediction = np.array(testprediction.to('cpu'))
                testprediction = pd.DataFrame(testprediction, columns=self.OutputColumns)

                testy = np.array(testy.to('cpu'))
                testy = pd.DataFrame(testy, columns=self.OutputColumns)

        return testprediction, testy

    def Predict(self, DFinput):
        # Normalize
        if self.normalizetype == 'minmax':
            DFinput = (DFinput - self.DFmin[self.InputColumns]) / (
                        self.DFmax[self.InputColumns] - self.DFmin[self.InputColumns])
        elif self.normalizetype == 'gaussian':
            DFinput = (DFinput - self.DFmean[self.InputColumns]) / self.DFstd[self.InputColumns]

        # Casting
        Input = DFinput.to_numpy()
        Input = torch.FloatTensor(Input).to(self.device)

        # Predict
        self.SetMode('eval')
        with torch.no_grad():
            Output = self.forward(Input).to('cpu')
            Output = Output.numpy()
            Output = pd.DataFrame(Output, columns=self.OutputColumns)

        # Inverse Normalize
        if self.normalizetype == 'minmax':
            Output = Output * (self.DFmax[self.OutputColumns] - self.DFmin[self.OutputColumns]) + self.DFmin[
                self.OutputColumns]
        elif self.normalizetype == 'gaussian':
            Output = Output * self.DFstd[self.OutputColumns] + self.DFmean[self.OutputColumns]

        return Output

    def Describe(self):
        print("==========================================================")
        print("==========================================================")
        print(f"{self.NumInfeatures} Input Columns: {self.InputColumns}")
        print(f"{self.NumOutfeatures} Output Columns: {self.OutputColumns}")
        print(f"Epochs: {self.Numepochs}")
        print(f"Nodes: {self.Numfcnodes}")
        print(f"FC-hidden layers: {self.Numfchiddenlayers}")
        print(f"Train batch size: {self.Numbatchsize}")
        print(f"Learning rate: {self.learningrate}")
        print(f"Use fcbias in layers: {self.fcbias}")
        print(f"Shuffle train dataloader: {self.shuffle}")
        print(f"Activation function: {self.fcactivationtype.upper()}")
        print(f"Normalization type: {self.normalizetype.upper()}")
        print(f"Optimizer type: {self.optimizertype.upper()}")
        print(f"Loss function: {self.lossfunctype.upper()}")
        print(f"Train data rate: {self.traindatarate}")
        print(f"Validation data rate: {self.validdatarate}")
        print("==========================================================")
        print("==========================================================")










if __name__ == '__main__':
    quit()


