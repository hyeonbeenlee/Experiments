import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
    
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        # x.shape -> (batchsize,width,x,y)
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x) # 2D RFFT: (batchsize,width,x,y//2+1), drops negative frequency terms of the last dim
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1) # Left Upper Corner of 2D RFFT grid == Lower frequencies
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2) # Left Lower Corner of 2D RFFT grid == Lower frequencies
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) # the last dimension in s should be compressed dimension
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.p = nn.Linear(12, self.width)  # (N,12)->(N,width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)  # (N,width)->(N,width)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)  # (N,width)->(N,width)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)  # (N,width)->(N,width)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)  # (N,width)->(N,width)
        self.mlp0 = MLP(self.width, self.width, self.width)  # (N,width)->(N,width)->(N,width), Conv2D(in,out,mid,1)
        self.mlp1 = MLP(self.width, self.width, self.width)  # (N,width)->(N,width)->(N,width), Conv2D(in,out,mid,1)
        self.mlp2 = MLP(self.width, self.width, self.width)  # (N,width)->(N,width)->(N,width), Conv2D(in,out,mid,1)
        self.mlp3 = MLP(self.width, self.width, self.width)  # (N,width)->(N,width)->(N,width), Conv2D(in,out,mid,1)
        self.w0 = nn.Conv2d(self.width, self.width, 1)  # (N,width) -> (N,width)
        self.w1 = nn.Conv2d(self.width, self.width, 1)  # (N,width) -> (N,width)
        self.w2 = nn.Conv2d(self.width, self.width, 1)  # (N,width) -> (N,width)
        self.w3 = nn.Conv2d(self.width, self.width, 1)  # (N,width) -> (N,width)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)
    
    def forward(self, x):
        # x.shape -> (batchsize,x,y,t_in)
        grid = self.get_grid(x.shape, x.device) # Get positional encoding (batchsize,x,y,2)
        x = torch.cat((x, grid), dim=-1)  # Concatenate raw_input + pos_encoding along time (batchsize,x,y,t_in+2)
        x = self.p(x)  # nn.Linear(12, width) along the last axis: (batchsize,x,y,width)
        x = x.permute(0, 3, 1, 2) # Permute (batchsize,width,x,y)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        
        # Start of a Fourier Layer
        x1 = self.norm(self.conv0(self.norm(x))) # Shape does not change from InstanceNorm/SpectralConv2d ops (batchsize,width,x,y)
        x1 = self.mlp0(x1) # Conv2D MultiLayer, (N,C,H,W) -> (N,Cout,Hout,Wout) but shape doesn't change since kernel_size=1
        x2 = self.w0(x) # Pointwise linear transform (Conv2D), (N,C,H,W) -> (N,Cout,Hout,Wout) but shape doesn't change since kernel_size=1
        x = x1 + x2 # Residual connection sum
        x = F.gelu(x) # Nonlinearity pass (batchsize,width,x,y)
        # End of a Fourier Layer
        
        # Repeated Fourier MultiLayers and nonlinearity
        x1 = self.norm(self.conv1(self.norm(x))) # Repeated
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # Repeated Fourier MultiLayers and nonlinearity
        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # Repeated Fourier MultiLayers and nonlinearity
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x) # q = MLP(in_channel=width, out_channel=1, mid_channel=width * 4),  (batchsize,width,x,y) -> (batchsize,4*width,x,y) -> (batchsizd,1,x,y)
        x = x.permute(0, 2, 3, 1)  # (batchsize,x,y,1) ==> batched single timestep 2D grid
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)  # (size_x) vector
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  # (N,size_x,1,1) tensor, dim(size_x)=linspace(0,1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)  # (size_y) vector
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])  # (N,1,size_y,1) tensor, dim(size_x)=linspace(0,1)
        return torch.cat((gridx, gridy), dim=-1).to(device)  # Concatenate along time, (t-10,t-9,...t-1, [x,y])

batchsize = 32
grids_x = 64
grids_y = 64
t_in = 10
fourier_modes = 12
width = 20

fno = FNO2d(fourier_modes, fourier_modes, width)
p = nn.Linear(t_in + 2, width)

# Forward pass analysis
x0 = torch.rand(batchsize, grids_x, grids_y, t_in)  # Batched input from dataloader (N,x,y,t_in)
try:
    fno(x0)
except:
    raise ValueError("Improper input shape")
grid = get_grid(x0.shape, x0.device)  # Positional encoding (N,x,y,2)
x1 = torch.cat([x0, grid], axis=-1)  # Raw input + pos encoding (N,x,y,t_in+2)
x2 = p(x1)  # Linear transform P is applied to the last dimension (N,x,y,width)
x2 = x2.permute(0, 3, 1, 2)  # Permuted to (N,width,x,y)

pass
