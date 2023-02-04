import ModuleCompositeNN as C
from ModuleFCNN import FCNN
import torch
import matplotlib.pyplot as plt
import DataProcessing as DP
import pandas as pd
from sklearn.metrics import r2_score as R2
from torch.utils.data import TensorDataset,DataLoader

def f_yl(x):
	y = 0.5 * (6 * x - 2) ** 2 * torch.sin(12 * x - 4) + 10 * (x - 0.5) - 5
	return y

def f_dyl(x):
	dy = 6 * (6 * x - 2) * torch.sin(12 * x - 4) + 6 * (6 * x - 2) ** 2 * torch.cos(12 * x - 4) + 10
	return dy

def f_yh(x):
	y = (6 * x - 2) ** 2 * torch.sin(12 * x - 4)
	return y

# TrainData
x_lo = torch.linspace(0, 1, 11).view(-1, 1)
y_lo = f_yl(x_lo)
dy_lo = f_dyl(x_lo)

x_hi = torch.tensor([0, 0.4, 0.6, 1]).view(-1, 1)
y_hi = f_yh(x_hi)

# TestData
test_x = torch.linspace(0, 1, 1001).view(-1, 1)
test_yl = f_yl(test_x)
test_yh = f_yh(test_x)

data_lo = pd.DataFrame(torch.cat([x_lo, y_lo, dy_lo], dim=1).numpy(), columns=['x', 'yl', 'dyl'])
data_hi = pd.DataFrame(torch.cat([x_hi, y_hi], dim=1).numpy(), columns=['x', 'yh'])
testdata = pd.DataFrame(torch.cat([test_x, test_yl, test_yh], dim=1).numpy(), columns=['x', 'yl', 'yh'])

dl_lo=DataLoader(TensorDataset(x_lo,y_lo,dy_lo),batch_size=x_lo.shape[0],shuffle=True)
dl_hi=DataLoader(TensorDataset(x_hi,y_hi),batch_size=x_hi.shape[0],shuffle=True)

# HF only
NN_h2 = FCNN()
NN_h2.BaseDataset(data_hi, inputcols=['x'], outputcols=['yh'])
NN_h2.SetHyperParams(epochs=10000, nodes=20, hiddenlayers=4, activation='tanh', param_init=True, loss='mse')
NN_h2.SetDataloader(normalizetype='none')
NN_h2.CPU()

optimizer=torch.optim.Adam(params=[{'params':NN_h2.parameters()}],lr=1e-3)
Epochs=10000
earlystopR2=0.999

for epoch in range(Epochs):
	for batch in dl_hi:
		batch_x_hi,batch_y_hi=batch
		
		pred_y_hi=NN_h2.forward(batch_x_hi)
		loss=torch.mean(torch.square(pred_y_hi-batch_y_hi))
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	R2value = R2(batch_y_hi.detach().numpy(), pred_y_hi.detach().numpy())
	if (epoch+1)%500==0:
		print(f"Epoch {epoch+1}, Loss={loss.item():.10f}, R2={R2value:.5f}")
	if R2value>=earlystopR2:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.10f}, R2={R2value:.5f}")
		break
		
prediction_yh_HF = NN_h2.Predict(test_x)

DP.MyPlotTemplate()
Fig = plt.figure(figsize=(10, 9))
ax = Fig.add_subplot(2, 2, 1)
ax.plot(test_x, test_yl, linestyle='--', color='black')
ax.plot(test_x, test_yh, color='black')
ax.scatter(x_lo, y_lo, marker='o', facecolors='none', edgecolors='blue', s=100, linewidth=2, label='Low-fidelity training data')
ax.scatter(x_hi, y_hi, marker='x', c='red', s=100, linewidth=2, label='High-fidelity training data')
ax.plot(test_x, prediction_yh_HF, color='violet', linewidth=3, label='High-fidelity learned')
ax.grid()
ax.legend(loc=2)
ax.set(xlim=(0, 1), ylim=(-10, 20), xlabel='$x$', ylabel='$y$')


# Low-fidelity
NN_l = FCNN()
NN_l.BaseDataset(data_lo, inputcols=['x'], outputcols=['yl'])
NN_l.SetHyperParams(epochs=10000, nodes=10, hiddenlayers=2, activation='tanh', param_init=True)
NN_l.SetDataloader(normalizetype='none')
NN_l.CPU()

optimizer = torch.optim.Adam(params=[{'params': NN_l.parameters()}], lr=1e-3)
Epochs = 100000
earlystopR2=0.999
l2_lambda = 0.01

for epoch in range(Epochs):
	for batch in dl_lo:
		batch_x_lo,batch_y_lo,batch_dy_lo=batch
		batch_x_lo.requires_grad = True
		
		pred_y_lo = NN_l.forward(batch_x_lo)
		pred_y_lo_grad, = torch.autograd.grad(outputs=pred_y_lo, inputs=batch_x_lo, grad_outputs=torch.ones_like(pred_y_lo),
											  retain_graph=True, create_graph=True, only_inputs=True)
		loss = torch.mean(torch.square(pred_y_lo - batch_y_lo)) #+ torch.mean(torch.square(pred_y_lo_grad - batch_dy_lo))
	
		l2_reg = 0
		# for param in NN_l.parameters():
		# 	l2_reg+=torch.linalg.norm(param)**2
		loss += l2_lambda * l2_reg
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	R2value = R2(batch_y_lo.detach().numpy(), pred_y_lo.detach().numpy())
	if (epoch + 1) % 500 == 0:
		print(f"Epoch {epoch+1}, Loss={loss.item():.10f}, R2={R2value:.5f}")
	if R2value>=earlystopR2:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.10f}, R2={R2value:.5f}")
		break

prediction_yl = NN_l.Predict(test_x)

ax = Fig.add_subplot(2, 2, 2)
ax.plot(test_x, test_yl, linestyle='--', color='black')
ax.plot(test_x, test_yh, color='black')
ax.scatter(x_lo, y_lo, marker='o', facecolors='none', edgecolors='blue', s=100, linewidth=2, label='Low-fidelity training data')
ax.scatter(x_hi, y_hi, marker='x', c='red', s=100, linewidth=2, label='High-fidelity training data')
ax.plot(test_x, prediction_yl, label='Low-fidelity learned', color='green', linewidth=3)
ax.grid()
ax.legend(loc=2)
ax.set(xlim=(0, 1), ylim=(-10, 20), xlabel='$x$', ylabel='$y$')

plt.show()

# Multi-fidelity
Epochs = 10000
l2_lambda = 0.01

NN_l = C.StackedNN(1, 2, 20, 1)
NN_h1 = C.StackedNN(2, 1, 10, 1, activation='none')
NN_h2 = C.StackedNN(2, 2, 10, 1)

Par = C.LearnableParams()
Par.RegisterAlpha(init_value=0.5)

optimizer = torch.optim.Adam(params=[{'params': NN_l.parameters()},
									 {'params': NN_h1.parameters()},
									 {'params': NN_h2.parameters()},
									 {'params': Par.parameters()}], lr=1e-3)

x_lo.requires_grad = True
for epoch in range(Epochs):
	pred_y_lo = NN_l.forward(x_lo)  # pred_y_lo = pred_y_lo 이므로 우변 미분 시 d(pred_y_lo)/d(pred_y_lo)=torch.ones_like(pred_y_lo) 가 된다
	pred_y_lo_grad, = torch.autograd.grad(outputs=pred_y_lo, inputs=x_lo, grad_outputs=torch.ones_like(pred_y_lo),
										  retain_graph=True, create_graph=True, only_inputs=True)
	mse_yl = torch.mean(torch.square(pred_y_lo - y_lo)) + torch.mean(torch.square(pred_y_lo_grad - dy_lo))
	
	pred_y_lo_hi = NN_l.forward(x_hi)
	pred_y_hi = NN_h1.forward(torch.cat([x_hi, pred_y_lo_hi], dim=1)) * (Par.alpha)
	pred_y_hi += NN_h2.forward(torch.cat([x_hi, pred_y_lo_hi], dim=1)) * (1 - Par.alpha)
	mse_yh = torch.mean(torch.square(pred_y_hi - y_hi))
	
	l2_reg = 0
	for param in NN_l.parameters():
		l2_reg += torch.linalg.norm(param) ** 2
	for param in NN_h2.parameters():
		l2_reg += torch.linalg.norm(param) ** 2
	l2_reg += torch.linalg.norm(Par.alpha) ** 2
	
	loss = mse_yl + mse_yh + l2_lambda * l2_reg
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if (epoch + 1) % 500 == 0:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.5f}, Alpha={Par.alpha.item():.5f}, MSE_yl={mse_yl.item():.5f}, MSE_yh={mse_yh.item():.5f}, L2_reg={(l2_lambda * l2_reg).item():.5f}")
x_lo.requires_grad = False

prediction_yl = NN_l.forward(test_x)
prediction_yh = NN_h1.forward(torch.cat([test_x, prediction_yl], dim=1)) * (Par.alpha)
prediction_yh += NN_h2.forward(torch.cat([test_x, prediction_yl], dim=1)) * (1 - Par.alpha)

prediction_yl = prediction_yl.detach()
prediction_yh = prediction_yh.detach()

ax = Fig.add_subplot(2, 2, 3)
ax.plot(test_x, test_yl, linestyle='--', color='black')
ax.plot(test_x, test_yh, color='black')
ax.scatter(x_lo, y_lo, marker='o', facecolors='none', edgecolors='blue', s=100, linewidth=2, label='Low-fidelity training data')
ax.scatter(x_hi, y_hi, marker='x', c='red', s=100, linewidth=2, label='High-fidelity training data')
ax.plot(test_x, prediction_yh, label='Multi-fidelity learned CNN', color='gold', linewidth=3)
ax.grid()
ax.legend(loc=2)
ax.set(xlim=(0, 1), ylim=(-10, 20), xlabel='$x$', ylabel='$y$')
Fig.tight_layout()
plt.show()
