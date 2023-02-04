import torch
import glob
import pandas as pd
import os
from RNN import MyRNNClass

Data=pd.read_csv("RecurDyn\Data.csv",index_col=0)
InputCols=['Pos1_Relative@RevJoint1', 'Pos1_Relative@RevJoint2',
       'Vel1_Relative@RevJoint1', 'Vel1_Relative@RevJoint2',
       'Acc1_Relative@RevJoint1', 'Acc1_Relative@RevJoint2']
OutputCols=InputCols

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DNNModel = MyRNNClass()
DNNModel.FixSeed(777)
DNNModel = DNNModel.to(device)
DNNModel.UseDataframe(Data,inputcols=InputCols,outputcols=OutputCols)
DNNModel.SetHyperParams(epochs=1,fcnodes=128,fchiddenlayers=3,batchsize=64,lr=1e-3,fcbias=True,fcactivation='relu',loss='mse',optimizer='adam',
                        recurrent_type='lstm',recurrent_hiddensize=2,recurrent_layers=3,recurrent_bias=True,recurrent_nonlinearity='tanh',recurrent_dropout=0.01)