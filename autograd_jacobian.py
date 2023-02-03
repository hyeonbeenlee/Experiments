import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import MyPackage as mp

def f(t, x):
    return torch.cat([0.1 * x ** 2 + 0.5 * t ** 2, torch.zeros_like(t), torch.zeros_like(t)], dim=1)

def df_dt(t, x):
    return torch.cat([t, torch.zeros_like(t), torch.zeros_like(t)], dim=1)

def df2_dt2(t, x):
    return torch.cat([torch.ones_like(t), torch.zeros_like(t), torch.zeros_like(t)], dim=1)

class TestNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TestNet, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, 50)
        self.linear2 = nn.Linear(50, output_dim)
        
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x

net = TestNet(2, 3).cuda()
batchsize = 64
t = torch.linspace(0, 5, batchsize).unsqueeze(1).cuda()
t = t.requires_grad_()
x = torch.rand(batchsize, 1).cuda()
X = torch.cat([t, x], dim=1)
optim = torch.optim.RAdam(net.parameters(), lr=1e-2)

loss = torch.inf
epoch = 0
while loss > 1e-3:
    epoch += 1
    label = f(t.detach(), x)
    # Forward
    pred = net(X)
    # Loss
    loss = torch.mean(torch.square(label - pred))
    # First order derivative loss
    pred_grad, = torch.autograd.grad(pred, t, torch.ones_like(pred), create_graph=True, retain_graph=True)
    loss+= torch.mean(torch.square(pred_grad - df_dt(t.detach(), x)))
    # Zero out grad
    for p in net.parameters():
        p.grad = None
    # Backward
    loss.backward()
    optim.step()
    # Print
    if epoch % 100 == 0:
        print(label[0], pred[0])
        print(label[-1], pred[-1])
        print(loss.item())

# Autograd tracks forward comp
t.grad.zero_() # init
pred = net(X)
label = f(t.detach(), x)

# Computing first and second order derivative wrt time

# Solution 1
grad_net1, = torch.autograd.grad(pred[:,0], t, torch.ones_like(pred[:,0]), create_graph=True)
grad_net2, = torch.autograd.grad(grad_net1, t, torch.ones_like(grad_net1))

# Solution 2
# pred.backward(gradient=torch.ones_like(pred), create_graph=True)
# grad_net1 = t.grad
# grad_net2 = None  # inaccessible through 'backward()' call

grad1_label = df_dt(t, x)
grad2_label = df2_dt2(t, x)

with torch.no_grad():
    mp.visualize.PlotTemplate(15)
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(pred.cpu())
    axes[0].plot(label.cpu())
    axes[1].plot(grad_net1.cpu())
    axes[1].plot(grad1_label.cpu())
    axes[2].plot(grad2_label.cpu())
    if grad_net2 is not None:
        axes[2].plot(grad_net2.cpu())
    plt.show()
