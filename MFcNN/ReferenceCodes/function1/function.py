import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim, autograd
from torch.nn import functional as F

torch.manual_seed(1234)
np.random.seed(1234)

class Unit(nn.Module):

    def __init__(self, in_N, out_N):
        super(Unit, self).__init__()
        self.in_N = in_N
        self.out_N = out_N
        self.L = nn.Linear(in_N, out_N)

    def forward(self, x):
        x1 = self.L(x)
        x2 = torch.tanh(x1)
        return x2


class NN1(nn.Module):

    def __init__(self, in_N, width, depth, out_N):
        super(NN1, self).__init__()
        self.width = width
        self.in_N = in_N
        self.out_N = out_N
        self.stack = nn.ModuleList()

        self.stack.append(Unit(in_N, width))

        for i in range(depth):
            self.stack.append(Unit(width, width))

        self.stack.append(nn.Linear(width, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


class NN2(nn.Module):
    def __init__(self, in_N, width, depth, out_N):
        super(NN2, self).__init__()
        self.in_N = in_N
        self.width = width
        self.depth = depth
        self.out_N = out_N

        self.stack = nn.ModuleList()

        self.stack.append(nn.Linear(in_N, width))

        for i in range(depth):
            self.stack.append(nn.Linear(width, width))

        self.stack.append(nn.Linear(width, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def main():
    # Reference data 생성
    x_hi = np.array([0, 0.4, 0.6, 1]).reshape(-1, 1)
    x_lo = np.linspace(0, 1, 11).reshape(-1, 1)
    y_hi_star = (6 * x_hi - 2) ** 2 * np.sin(12 * x_hi - 4)
    x = np.linspace(0, 1, 101).reshape(-1, 1)
    y_hi = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    y_lo = 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5
    y_lo_star = 0.5 * (6 * x_lo - 2) ** 2 * np.sin(12 * x_lo - 4) + 10 * (x_lo - 0.5) - 5
    y_lo_star_prime = 10 + 6 * (6 * x_lo - 2) * np.sin(12 * x_lo - 4) + \
                      0.5 * (6 * x_lo - 2) ** 2 * np.cos(12 * x_lo - 4) * 12
    
    fig, ax = plt.subplots()
    ax.plot(x, y_hi, label='$y_H$', color='black')
    ax.plot(x, y_lo, label='$y_L$', color='black', linestyle='dashed')
    ax.scatter(x_hi, y_hi_star,marker='x', color='red', linewidth='2',  label='high-fidelity training data')
    ax.scatter(x_lo, y_lo_star, marker='o', color='blue', edgecolors='blue', label='low-fidelity training data')
    ax.set(xlabel='x', ylabel='y')
    ax.legend()
    plt.show()

    # High fidelity data 예측 NNh2
    in_N = 1
    width = 10
    depth = 2
    out_N = 1
    model_h = NN1(in_N, width, depth, out_N)
    model_h.apply(weights_init)
    optimizer = optim.Adam(model_h.parameters(), lr=0.001)
    nIter = 2000
    it = 0
    loss_value = 1
    while loss_value > 1e-3:
        pred_h = model_h(torch.from_numpy(x_hi).float())
        loss = torch.mean(torch.square(pred_h - torch.from_numpy(y_hi_star).float()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if it % 50 == 0:
            print('It:', it, 'Loss', loss.item())
        it = it + 1
    nn_pred_h = model_h(torch.from_numpy(x).float())

    fig2, ax2 = plt.subplots()
    line = ax2.plot(x, nn_pred_h.detach().numpy(), label='DNN though HF', color='darkviolet')
    line[0].set_dashes([2, 2, 4, 2])  # 2pt line, 2pt break, 4pt line, 2pt break
    ax2.plot(x, y_hi, label='$Exact$', color='black')
    ax2.scatter(x_hi, y_hi_star, marker='x', color='red', linewidth=2)
    ax2.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    # Low fidelity data 예측 NNl
    model_h = NN1(1, 20, 4, 1)
    model_h.apply(weights_init)
    optimizer = optim.Adam(model_h.parameters(), lr=1e-3)
    loss_value = 1
    x_lo_r = torch.from_numpy(x_lo).float() #
    x_lo_r.requires_grad_()
    it = 0
    while loss_value > 1e-3:
        pred_h = model_h(x_lo_r)
        grads = autograd.grad(outputs=pred_h, inputs=x_lo_r,
                              grad_outputs=torch.ones_like(pred_h),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss = torch.mean(torch.square(pred_h - torch.from_numpy(y_lo_star).float())) + \
               torch.mean(torch.sum(torch.square(grads - torch.from_numpy(y_lo_star_prime).float()), 1, keepdim=True))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if it % 50 == 0:
            print('It:', it, 'Loss', loss.item())
        it = it + 1
    nn_pred_h = model_h(torch.from_numpy(x).float())
    fig3, ax3 = plt.subplots()
    line = ax3.plot(x, nn_pred_h.detach().numpy(), label='DNN though LF', color='darkviolet')
    line[0].set_dashes([2, 2, 4, 2])  # 2pt line, 2pt break, 4pt line, 2pt break
    ax3.plot(x, y_lo, label='$Exact$', color='black')
    ax3.scatter(x_lo, y_lo_star, marker='x', color='red', linewidth=2)
    ax3.legend()
    plt.show()
    
    
    
    
    
    # Multi fidelity data 예측
    # NNl + NNh1 + NNh2(No activation)
    # model_h + model3 + model4(No activation)
    alpha = torch.tensor([0.5])
    model3 = NN1(2, 20, 4, 1)
    model4 = NN2(2, 10, 2, 1)
    model3.apply(weights_init)
    model4.apply(weights_init)
    # 새로운 optimizer에 모든 파라미터 할당
    optimizer2 = optim.Adam([{'params': model3.parameters(), 'weight_decay': 0.01},
                             {'params': model_h.parameters(), 'weight_decay': 0.01},
                             {'params': model4.parameters(), 'weight_decay': 0.01},
                             {'params': alpha}], lr=1e-3)
    print(optimizer2.param_groups)
    quit()
    nIter2 = 10000
    x_lo_r = torch.from_numpy(x_lo).float() #lofi x
    x_lo_r.requires_grad_() #lofi x
    loss2_value = 1
    it = 0
    # x_lo_r.requires_grad_()
    while loss2_value > 1e-3 and it < nIter2:
        pred_h = model_h(x_lo_r)
        grads = autograd.grad(outputs=pred_h, inputs=x_lo_r,
                              grad_outputs=torch.ones_like(pred_h),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        # Loss1 = MSE(Label, NNl(x_lo)) + MSE(y_lo,grad(NNl(x_lo)))
        loss3 = torch.mean(torch.square(pred_h - torch.from_numpy(y_lo_star).float())) + \
               torch.mean(torch.sum(torch.square(grads - torch.from_numpy(y_lo_star_prime).float()), 1, keepdim=True))
        
        
        
        # NNl(x_hi)
        pred_2h = model_h(torch.from_numpy(x_hi).float())
        # alpha*NNh1(x_hi,NNl(x_hi))+(1-alpha)*NNh2(x_hi,NNl(x_hi))
        pred_2 = alpha * model3(torch.cat((torch.from_numpy(x_hi).float(), pred_2h), 1)) + \
                 (1 - alpha) * model4(torch.cat((torch.from_numpy(x_hi).float(), pred_2h), 1))
        # Loss2 = Loss1 + MSE(alpha*NNh1(x_hi,NNl(x_hi))+(1-alpha)*NNh2(x_hi,NNl(x_hi)), y_hi)
        loss2 = torch.mean(torch.square(pred_2 - torch.from_numpy(y_hi_star).float())) + loss3
        loss2_value = loss2.item()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        if it % 100 == 0:
            print('It:', it, 'Loss:', loss2.item())
            print(f'Alpha value={alpha}')
        it = it + 1
    xx_lo = model_h(torch.from_numpy(x).float())
    xx_high = alpha * model3(torch.cat((torch.from_numpy(x).float(), xx_lo), 1)) +\
              (1 - alpha) * model4(torch.cat((torch.from_numpy(x).float(), xx_lo), 1))
    print(alpha)
    fig4, ax4 = plt.subplots()
    ax4.plot(x, xx_high.detach().numpy(), label='DNN though multi-fidelity model', color='darkviolet')
    ax4.plot(x, y_hi, label='$Exact$', color='black')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
