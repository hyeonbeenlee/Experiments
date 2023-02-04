import torch
import matplotlib.pyplot as plt
import MyPackage as mp

testx = torch.zeros(64, 64)
x=torch.linspace(0, 2 * torch.pi, testx.shape[0])
for j in range(testx.shape[1]):
    testx[:,j] += 10*torch.sin(1*x)+20*torch.cos(5*x)
    testx[j,:] += 10*torch.sin(1*x)+20*torch.cos(5*x)

mp.visualize.PlotTemplate()
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].matshow(testx)
ax[1].matshow(torch.fft.fft2(testx).real)
ax[2].matshow(torch.fft.rfft2(testx).real)
fig.tight_layout()
plt.show()
