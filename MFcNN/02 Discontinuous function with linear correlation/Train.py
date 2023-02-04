import ModuleCompositeNN as C
import torch
import matplotlib.pyplot as plt
import DataProcessing as DP

def f_yl1(x):
	y = 0.5 * (6 * x - 2) ** 2 * torch.sin(12 * x - 4) + 10 * (x - 0.5) - 5
	return y

def f_yl2(x):
	y = 3 + 0.5 * (6 * x - 2) ** 2 * torch.sin(12 * x - 4) + 10 * (x - 0.5) - 5
	return y

def f_yl(x):
	y = torch.where(x <= 0.5, f_yl1(x), torch.zeros(1))
	y += torch.where(0.5 < x, f_yl2(x), torch.zeros(1))
	return y

def f_dyl1(x):
	dy = 6 * (6 * x - 2) * torch.sin(12 * x - 4) + 0.5 * (6 * x - 2) ** 2 * torch.cos(12 * x - 4) * 12 + 10
	return dy

def f_dyl2(x):
	dy = 6 * (6 * x - 2) * torch.sin(12 * x - 4) + 0.5 * (6 * x - 2) ** 2 * torch.cos(12 * x - 4) * 12 + 10
	return dy

def f_dyl(x):
	dy = torch.where(x <= 0.5, f_dyl1(x), torch.zeros(1))
	dy += torch.where(0.5 < x, f_dyl2(x), torch.zeros(1))
	return dy

def f_yh1(x):
	y = 2 * f_yl(x) - 20 * x + 20
	return y

def f_yh2(x):
	y = 4 + 2 * f_yl(x) - 20 * x + 20
	return y

def f_yh(x):
	y = torch.where(x <= 0.5, f_yh1(x), torch.zeros(1))
	y += torch.where(0.5 < x, f_yh2(x), torch.zeros(1))
	return y

# TrainData
x_lo1 = torch.linspace(0, 0.4, 9)
x_lo2 = torch.linspace(0.6, 1, 9)
x_lo3 = torch.linspace(0.4, 0.6, 22)

x_lo = torch.sort(torch.cat([x_lo1[:-1],x_lo2[1:], x_lo3])).values.view(-1, 1)
y_lo = f_yl(x_lo)
dy_lo = f_dyl(x_lo)

x_hi = torch.tensor([0.2, 0.4, 0.6, 0.75, 0.9]).view(-1, 1)
y_hi = f_yh(x_hi)

# TestData
test_x = torch.linspace(0, 1, 1001).view(-1, 1)
test_yl = f_yl(test_x)
test_yh = f_yh(test_x)




# High-fidelity
NN_h2 = C.StackedNN(1, 4, 20, 1)
optimizer = torch.optim.Adam(params=[{'params': NN_h2.parameters()}], lr=1e-3)
l2_lambda=0.01
Epochs = 3000
for epoch in range(Epochs):
	
	pred_y_hi = NN_h2.forward(x_hi)
	loss = torch.mean(torch.square(pred_y_hi - y_hi))
	
	l2_reg = 0
	# for param in NN_h2.parameters():
	# 	l2_reg += torch.linalg.norm(param) ** 2
	loss += l2_lambda * l2_reg
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if (epoch + 1) % 500 == 0:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.10f}")

prediction_yh_HF = NN_h2.forward(test_x).detach()

DP.MyPlotTemplate()
Fig = plt.figure(figsize=(10, 9))
ax = Fig.add_subplot(2, 2, 1)
ax.plot(test_x, test_yl, linestyle='--', color='black')
ax.plot(test_x, test_yh, color='black')
ax.scatter(x_lo, y_lo, marker='o',facecolors='none',edgecolors='blue', s=100,linewidth=2, label='Low-fidelity training data')
ax.scatter(x_hi, y_hi, marker='x', c='red', s=100,linewidth=2, label='High-fidelity training data')
ax.plot(test_x, prediction_yh_HF, color='violet', linewidth=3, label='High-fidelity learned')
ax.grid()
ax.legend(loc=2)
ax.set(xlim=(0, 1), ylim=(-10, 30), xlabel='$x$', ylabel='$y$')






# Low-fidelity
NN_l=C.StackedNN(1,2,10,1)
optimizer = torch.optim.Adam(params=[{'params': NN_l.parameters()}], lr=1e-3)
Epochs=15000
l2_lambda = 0.01

x_lo.requires_grad=True
for epoch in range(Epochs):
	
	pred_y_lo = NN_l.forward(x_lo)
	loss = torch.mean(torch.square(pred_y_lo - y_lo))
	
	l2_reg=0
	# for param in NN_l.parameters():
	# 	l2_reg+=torch.linalg.norm(param)**2
	loss+=l2_lambda*l2_reg
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if (epoch + 1) % 500 == 0:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.10f}")
x_lo.requires_grad = False

prediction_yl = NN_l.forward(test_x)
prediction_yl=prediction_yl.detach()

ax = Fig.add_subplot(2, 2, 2)
ax.plot(test_x, test_yl, linestyle='--', color='black')
ax.plot(test_x, test_yh, color='black')
ax.scatter(x_lo, y_lo, marker='o',facecolors='none',edgecolors='blue', s=100,linewidth=2, label='Low-fidelity training data')
ax.scatter(x_hi, y_hi, marker='x', c='red', s=100,linewidth=2, label='High-fidelity training data')
ax.plot(test_x, prediction_yl, label='Low-fidelity learned', color='green', linewidth=3)
ax.grid()
ax.legend(loc=2)
ax.set(xlim=(0, 1), ylim=(-10, 30), xlabel='$x$', ylabel='$y$')







# Multi-fidelity
Epochs = 25000
l2_lambda = 0.001

NN_l = C.StackedNN(1, 2, 10, 1)
NN_h1 = C.StackedNN(2, 1, 10, 1, activation='none')
NN_h2 = C.StackedNN(2, 4, 20, 1)

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
	mse_yl = torch.mean(torch.square(pred_y_lo - y_lo)) #+ torch.mean(torch.square(pred_y_lo_grad - dy_lo))
	
	pred_y_lo = NN_l.forward(x_hi)
	pred_y_hi = NN_h1.forward(torch.cat([x_hi, pred_y_lo], dim=1)) * (Par.alpha)
	pred_y_hi += NN_h2.forward(torch.cat([x_hi, pred_y_lo], dim=1)) * (1 - Par.alpha)
	mse_yh = torch.mean(torch.square(pred_y_hi - y_hi))
	
	l2_reg = 0
	# for param in NN_l.parameters():
	# 	l2_reg += torch.linalg.norm(param) ** 2
	# for param in NN_h2.parameters():
	# 	l2_reg += torch.linalg.norm(param) ** 2
	# l2_reg += torch.linalg.norm(Par.alpha) ** 2
	
	loss = mse_yl + mse_yh + l2_lambda * l2_reg
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if (epoch + 1) % 500 == 0:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.5f}, Alpha={Par.alpha.item():.5f}, MSE_yl={mse_yl.item():.5f}, MSE_yh={mse_yh.item():.5f}")
x_lo.requires_grad = False

prediction_yl = NN_l.forward(test_x)
prediction_yh = NN_h1.forward(torch.cat([test_x, prediction_yl], dim=1)) * (Par.alpha)
prediction_yh += NN_h2.forward(torch.cat([test_x, prediction_yl], dim=1)) * (1 - Par.alpha)

prediction_yl = prediction_yl.detach()
prediction_yh = prediction_yh.detach()

ax = Fig.add_subplot(2, 2, 3)
ax.plot(test_x, test_yl, linestyle='--', color='black')
ax.plot(test_x, test_yh, color='black')
ax.scatter(x_lo, y_lo, marker='o',facecolors='none',edgecolors='blue', s=100,linewidth=2, label='Low-fidelity training data')
ax.scatter(x_hi, y_hi, marker='x', c='red', s=100,linewidth=2, label='High-fidelity training data')
ax.plot(test_x, prediction_yh, label='Multi-fidelity learned CNN', color='gold', linewidth=3)
ax.grid()
ax.legend(loc=2)
ax.set(xlim=(0, 1), ylim=(-10, 30), xlabel='$x$', ylabel='$y$')
Fig.tight_layout()
plt.show()