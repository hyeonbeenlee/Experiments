import ModuleCompositeNN as C
import torch
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import DataProcessing as DP
from sklearn.metrics import r2_score as R2

def f_yh(x):
	y = torch.zeros((x.shape[0], 1))
	y+=(x[:,0].view(-1,1)-1)**2
	for i in range(1,20,1):
		y+=(2*x[:,i].view(-1,1)**2-x[:,i-1].view(-1,1))**2
	return y

def f_yl(x):
	y = 0.8 * f_yh(x) - 50
	for i in range(0,19,1):
		y-=0.4*x[:,i].view(-1,1)*x[:,i-1].view(-1,1)
	return y

# def f_dyl(x):
# 	dy=8*torch.pi*torch.cos(8*torch.pi*x)
# 	return dy

# TrainData
x_lo = torch.tensor(DP.LHCSampler(30000,20),dtype=torch.float32) * 6 - 3
y_lo = f_yl(x_lo)
print(f"Low fidelity train data ready.")

x_hi = torch.tensor(DP.LHCSampler(5000,20),dtype=torch.float32) * 6 - 3
y_hi = f_yh(x_hi)
print(f"High fidelity train data ready.")

# # TestData
test_xl = torch.tensor(DP.LHCSampler(10000,20),dtype=torch.float32) * 6 - 3
test_xh = torch.tensor(DP.LHCSampler(10000,20),dtype=torch.float32) * 6 - 3
test_yl = f_yl(test_xl)
test_yh = f_yh(test_xh)
print(f"Test data ready.")
# #
# # TensorDict={'x_lo':x_lo,'y_lo':y_lo,
# # 			'x_hi':x_hi,'y_hi':y_hi,
# # 			'test_xl':test_xl,'test_xh':test_xh,
# # 			'test_yl':test_yl,'test_yh':test_yh}
# # torch.save(TensorDict,'./TensorDict.torch')
# # quit()
#
# TensorDict=torch.load('./TensorDict.torch')
# x_lo=TensorDict['x_lo']
# y_lo=TensorDict['y_lo']
# x_hi=TensorDict['x_hi']
# y_hi=TensorDict['y_hi']
# test_xl=TensorDict['test_xl']
# test_xh=TensorDict['test_xh']
# test_yl=TensorDict['test_yl']
# test_yh=TensorDict['test_yh']



x_lo_min=x_lo.min(dim=0,keepdim=True)[0]
x_lo_max=x_lo.max(dim=0,keepdim=True)[0]
x_lo_mean=x_lo.mean(dim=0,keepdim=True)[0]
x_lo_std=x_lo.std(dim=0,keepdim=True)[0]
y_lo_min=y_lo.min(dim=0,keepdim=True)[0]
y_lo_max=y_lo.max(dim=0,keepdim=True)[0]
y_lo_mean=y_lo.mean(dim=0,keepdim=True)[0]
y_lo_std=y_lo.std(dim=0,keepdim=True)[0]
x_hi_min=x_hi.min(dim=0,keepdim=True)[0]
x_hi_max=x_hi.max(dim=0,keepdim=True)[0]
x_hi_mean=x_hi.mean(dim=0,keepdim=True)[0]
x_hi_std=x_hi.std(dim=0,keepdim=True)[0]
y_hi_min=y_hi.min(dim=0,keepdim=True)[0]
y_hi_max=y_hi.max(dim=0,keepdim=True)[0]
y_hi_mean=y_hi.mean(dim=0,keepdim=True)[0]
y_hi_std=y_hi.std(dim=0,keepdim=True)[0]


#최대최소 정규화
x_lo=(x_lo-x_lo_min)/(x_lo_max-x_lo_min)
x_hi=(x_hi-x_hi_min)/(x_hi_max-x_hi_min)
test_xl=(test_xl-x_lo_min)/(x_lo_max-x_lo_min)
test_xh=(test_xh-x_hi_min)/(x_hi_max-x_hi_min)
y_lo=(y_lo-y_lo_min)/(y_lo_max-y_lo_min)
y_hi=(y_hi-y_hi_min)/(y_hi_max-y_hi_min)
test_yl=(test_yl-y_lo_min)/(y_lo_max-y_lo_min)
test_yh=(test_yh-y_hi_min)/(y_hi_max-y_hi_min)

# Z 정규화
# x_lo=(x_lo-x_lo_mean)/(x_lo_std)
# x_hi=(x_hi-x_hi_mean)/(x_hi_std)
# test_xl=(test_xl-x_lo_mean)/(x_lo_std)
# test_xh=(test_xh-x_hi_mean)/(x_hi_std)
# y_lo=(y_lo-y_lo_mean)/(y_lo_std)
# y_hi=(y_hi-y_hi_mean)/(y_hi_std)
# test_yl=(test_yl-y_lo_mean)/(y_lo_std)
# test_yh=(test_yh-y_hi_mean)/(y_hi_std)



# GPU
x_lo=x_lo.cuda()
y_lo=y_lo.cuda()
x_hi=x_hi.cuda()
y_hi=y_hi.cuda()
test_xl=test_xl.cuda()
test_yl=test_yl.cuda()
test_xh=test_xh.cuda()
test_yh=test_yh.cuda()


# High-fidelity
NN_h2 = C.StackedNN(20, 4, 160, 1, activation='tanh').cuda()
optimizer = torch.optim.Adam(params=[{'params': NN_h2.parameters()}], lr=1e-3)
l2_lambda = 0.01
epoch=0
R2score=0
while R2score<0.99:
# while epoch<1000:
	pred_y_hi = NN_h2.forward(x_hi)
	loss = torch.mean(torch.square(pred_y_hi - y_hi))
	
	l2_reg = 0
	# for param in NN_h2.parameters():
	# 	l2_reg+=torch.linalg.norm(param)**2
	loss += l2_lambda * l2_reg
	
	R2score=R2(y_hi.detach().cpu(), pred_y_hi.detach().cpu())
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	epoch+=1
	
	if (epoch + 1) % 500 == 0:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.10f}")
		print(f"Train Data R2(y_H)={R2score:.6f}")
		print(f"Test Data R2(y_H)={R2(test_yh.detach().cpu(), NN_h2.forward(test_xh).detach().cpu()):.6f}")
		print()
print(f"Epoch {epoch + 1}, Loss={loss.item():.10f}")
print(f"Train Data R2(y_H)={R2score:.6f}")
print(f"Test Data R2(y_H)={R2(test_yh.detach().cpu(), NN_h2.forward(test_xh).detach().cpu()):.6f}")
print()

prediction_yh_HF = NN_h2.forward(test_xh).detach()

DP.MyPlotTemplate()
Fig = plt.figure(figsize=(10, 9))
ax = Fig.add_subplot(2, 2, 1)
ax.plot(test_yh.cpu(), test_yh.cpu(), c='black', linewidth=2, label='Exact')
ax.scatter(test_yh.cpu(), prediction_yh_HF.cpu(), marker='o', facecolors='none', edgecolors='violet', label='High-fidelity learned')
textbox = offsetbox.AnchoredText(f"$R^2$={R2(test_yh.cpu(),prediction_yh_HF.cpu()):.5f}", loc=4)
ax.add_artist(textbox)
ax.grid()
ax.legend(loc=2)
ax.set(xlabel='Exact $y_H$', ylabel='Predicted $y_H$')



# Low-fidelity
NN_l = C.StackedNN(20, 4, 128, 1, activation='tanh').cuda()
optimizer = torch.optim.Adam(params=[{'params': NN_l.parameters()}], lr=1e-3)
l2_lambda = 0.01
epoch=0
R2score=0

# x_lo.requires_grad = True
while R2score<0.99:
# while epoch < 20000:

	pred_y_lo = NN_l.forward(x_lo)
	# pred_y_lo_grad, = torch.autograd.grad(outputs=pred_y_lo, inputs=x_lo, grad_outputs=torch.ones_like(pred_y_lo),
	# 									  retain_graph=True, create_graph=True, only_inputs=True)
	loss = torch.mean(torch.square(pred_y_lo - y_lo)) #+ torch.mean(torch.square(pred_y_lo_grad - dy_lo))

	l2_reg = 0
	# for param in NN_l.parameters():
	# 	l2_reg+=torch.linalg.norm(param)**2

	loss += l2_lambda * l2_reg
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	epoch+=1
	R2score=R2(y_lo.detach().cpu(),pred_y_lo.detach().cpu())
	
	if (epoch + 1) % 500 == 0:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.10f}")
		print(f"Train Data R2(y_L)={R2score:.6f}")
		print(f"Test Data R2(y_L)={R2(test_yl.detach().cpu(), NN_l.forward(test_xl).detach().cpu()):.6f}")
		print()
print(f"Epoch {epoch + 1}, Loss={loss.item():.10f}")
print(f"Train Data R2(y_L)={R2score:.6f}")
print()
# x_lo.requires_grad = False

prediction_yl = NN_l.forward(test_xl)
prediction_yl = prediction_yl.detach()

ax = Fig.add_subplot(2, 2, 2)
ax.plot(test_yl.cpu(), test_yl.cpu(), c='black', linewidth=2, label='Exact')
ax.scatter(test_yl.cpu(), prediction_yl.cpu(), marker='o', facecolors='none', edgecolors='green', label='Low-fidelity learned')
textbox = offsetbox.AnchoredText(f"$R^2$={R2(test_yl.cpu(),prediction_yl.cpu()):.5f}", loc=4)
ax.add_artist(textbox)
ax.grid()
ax.legend(loc=2)
ax.set(xlabel='Exact $y_L$', ylabel='Predicted $y_L$')



# Multi-fidelity
epoch=0
R2score_L=0
R2score_H=0
l2_lambda = 0.01

NN_l = C.StackedNN(20, 4, 128, 1, activation='tanh').cuda()
NN_h1 = C.StackedNN(21, 1, 10, 1, activation='none').cuda()
NN_h2 = C.StackedNN(21, 2, 64, 1, activation='tanh').cuda()

Par = C.LearnableParams()
Par.RegisterAlpha(init_value=0.5)
Par.cuda()

optimizer = torch.optim.Adam(params=[{'params': NN_l.parameters()},
									 {'params': NN_h1.parameters()},
									 {'params': NN_h2.parameters()},
									 {'params': Par.parameters()}], lr=1e-3)

# x_lo.requires_grad = True
while R2score_L<0.99 or R2score_H<0.99:
# while epoch<1000:
	
	pred_y_lo = NN_l.forward(x_lo)  # pred_y_lo = pred_y_lo 이므로 우변 미분 시 d(pred_y_lo)/d(pred_y_lo)=torch.ones_like(pred_y_lo) 가 된다
	# pred_y_lo_grad, = torch.autograd.grad(outputs=pred_y_lo, inputs=x_lo, grad_outputs=torch.ones_like(pred_y_lo),
	# 									  retain_graph=True, create_graph=True, only_inputs=True)
	mse_yl = torch.mean(torch.square(pred_y_lo - y_lo)) #+ torch.mean(torch.square(pred_y_lo_grad - dy_lo))
	R2score_L = R2(y_lo.detach().cpu(), pred_y_lo.detach().cpu())
	
	pred_y_lo = NN_l.forward(x_hi)
	pred_y_hi = NN_h1.forward(torch.cat([x_hi, pred_y_lo], dim=1)) * (Par.alpha)
	pred_y_hi += NN_h2.forward(torch.cat([x_hi, pred_y_lo], dim=1)) * (1 - Par.alpha)
	mse_yh = torch.mean(torch.square(pred_y_hi - y_hi))
	R2score_H = R2(y_hi.detach().cpu(), pred_y_hi.detach().cpu())
	
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
	epoch+=1
	
	
	if (epoch + 1) % 500 == 0:
		print(f"Epoch {epoch + 1}, Loss={loss.item():.5f}, Alpha={Par.alpha.item():.5f}, MSE_yl={mse_yl.item():.5f}, MSE_yh={mse_yh.item():.5f}")
		print(f"Train Data R2(y_L)={R2score_L:.6f}")
		print(f"Train Data R2(y_H)={R2score_H:.6f}")
		print(f"Test Data R2(y_L)={R2(test_yl.detach().cpu(), NN_l.forward(test_xl).detach().cpu()):.6f}")
		print(f"Test Data R2(y_H)={R2(test_yh.detach().cpu(), (Par.alpha*NN_h1.forward(torch.cat((test_xh,NN_l.forward(test_xh)),dim=1))+((1-Par.alpha)*NN_h2.forward(torch.cat((test_xh,NN_l.forward(test_xh)),dim=1)))).detach().cpu()):.6f}")
		print()
print(f"Epoch {epoch + 1}, Loss={loss.item():.5f}, Alpha={Par.alpha.item():.5f}, MSE_yl={mse_yl.item():.5f}, MSE_yh={mse_yh.item():.5f}")
print(f"Train Data R2(y_L)={R2score_L:.6f}")
print(f"Train Data R2(y_H)={R2score_H:.6f}")
print()
# x_lo.requires_grad = False

prediction_yl = NN_l.forward(test_xh)
prediction_yh = NN_h1.forward(torch.cat([test_xh, prediction_yl], dim=1)) * (Par.alpha)
prediction_yh += NN_h2.forward(torch.cat([test_xh, prediction_yl], dim=1)) * (1 - Par.alpha)

prediction_yl = prediction_yl.detach()
prediction_yh = prediction_yh.detach()

ax = Fig.add_subplot(2, 2, 3)
ax.plot(test_yh.cpu(), test_yh.cpu(), c='black', linewidth=2, label='Exact')
ax.scatter(test_yh.cpu(), prediction_yh.cpu(), marker='o', facecolors='none', edgecolors='gold', label='Multi-fidelity learned CNN')
textbox = offsetbox.AnchoredText(f"$R^2$={R2(test_yh.cpu(),prediction_yh.cpu()):.5f}", loc=4)
ax.add_artist(textbox)
ax.grid()
ax.legend(loc=2)
ax.set(xlabel='Exact $y_H$', ylabel='Predicted $y_H$')
Fig.tight_layout()
plt.show()

ModelStateDict={"NN_l":NN_l.state_dict(),
				"NN_h1":NN_h1.state_dict(),
				"NN_h2":NN_h2.state_dict()}



torch.save(ModelStateDict,"./ModelStateDict.pt")
