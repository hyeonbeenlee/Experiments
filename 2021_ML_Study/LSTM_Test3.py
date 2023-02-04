import numpy as np
import torch
import matplotlib.pyplot as plt

#creating the dataset
x = np.arange(1,721,1)
y = np.sin(x*np.pi/180)  + np.random.randn(720)*0.05

X = []
Y = []
for i in range(0,710):
     list1 = []
     for j in range(i,i+10):
         list1.append(y[j])
     X.append(list1)
     Y.append(y[j+1])
     
#train test split
X = np.array(X)
Y = np.array(Y)
x_train = X[:360]
x_test = X[360:]
y_train = Y[:360]
y_test = Y[360:]


class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]
        
    def __getitem__(self, idx):
        #인덱싱 시 반환값 결정
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        #len함수 반환값 결정
        return self.len

dataset = timeseries(x_train,y_train)
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset,shuffle=True,batch_size=256)