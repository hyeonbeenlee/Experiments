import torch
import matplotlib.pyplot as plt
import MyPackage as mp
from torch import sin, cos, pi
import numpy as np

def fft1d_vector():
    # Set
    Fs = 2000  # Sampling frequency
    T = 1 / Fs  # Sampling time interval
    t_end = 0.5  # End time
    t = torch.arange(0, t_end, T)  # Time vector
    signal = 6 * sin(2 * pi * 60 * t) + 10 * cos(2 * pi * 120 * t)  # 60Hz(-pi/2 shifted) and 120Hz(non-shifted) cosine sum
    # signal += 0.1 * torch.randn(t.shape[0], )  # Small Gaussian noise
    n_samples = signal.shape[0]
    
    # Compute FFT
    f_fft = torch.arange(n_samples) * (Fs / n_samples)  # Frequency
    f_fft = f_fft[:n_samples // 2 + 1]  # Single sided frequency
    a_fft = torch.fft.fft(signal) / n_samples  # Normalize with factor of len(signal)
    a_fft = a_fft[:n_samples // 2 + 1]  # Single sided FFT: Drop negative frequency (imag part)
    a_fft *= 2  # Single sided FFT: Double amplitude (rewarding the imag part drop)
    phase_angles = torch.angle(a_fft) * 180 / pi
    
    # IFFT
    signal_ifft = torch.fft.irfft(a_fft) / 2 * n_samples  # Use iRfft since single sided.
    
    # Plot
    mp.visualize.PlotTemplate()
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    axes[0].plot(t, signal)  # Original singal
    axes[1].plot(f_fft, a_fft.real, label='Cosine Coefficients (Real)')  # Real signals FR: e^{i\theta}=cos(\theta)+i*sin(\theta)
    axes[1].plot(f_fft, a_fft.imag, label='Sine Coefficients (Imaginary)')  # Imag signals FR
    axes[1].plot(f_fft, torch.abs(a_fft), label='Absolute Coefficients (Re+Imag)')  # Absolute value
    axes[2].plot(phase_angles)  # Complex number phase angles
    axes[3].plot(t, signal_ifft)  # Inverse FFT
    
    mp.visualize.IncreaseLegendLinewidth(axes[1].legend(loc=4))
    fig.tight_layout()
    plt.show()

def fft1d_array():
    # Set
    Fs = 2000  # Sampling frequency
    T = 1 / Fs  # Sampling time interval
    t_end = 0.5  # End time
    t = torch.arange(0, t_end, T).reshape(-1, 1)  # Time vector
    signal = 6 * sin(2 * pi * 60 * t) + 10 * sin(2 * pi * 120 * t).reshape(-1, 1).repeat(1, 32)  # (N,32) singal array
    # signal += 0.1 * torch.randn(t.shape[0], )  # Small Gaussian noise
    n_samples = signal.shape[0]
    
    # Compute FFT
    f_fft = torch.arange(n_samples) * (Fs / n_samples)  # Frequency
    f_fft = f_fft[:n_samples // 2 + 1]  # Single sided frequency
    a_fft = torch.fft.fft(signal, axis=0) / n_samples  # Normalize with factor of len(signal)
    a_fft = a_fft[:n_samples // 2 + 1, ...]  # Single sided FFT: Drop negative frequency (imag part)
    a_fft *= 2  # Single sided FFT: Double amplitude (rewarding the imag part drop)
    phase_angles = torch.angle(a_fft) * 180 / pi
    
    # IFFT
    signal_ifft = torch.fft.irfft(a_fft, axis=0) / 2 * n_samples  # Use iRfft since single sided.
    
    # Plot
    mp.visualize.PlotTemplate()
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    axes[0].plot(t, signal)  # Original singal
    for i in range(32):
        axes[1].plot(f_fft, a_fft[..., i].real, label='Cosine Coefficients (Real)')  # Real signals FR: e^{i\theta}=cos(\theta)+i*sin(\theta)
        axes[1].plot(f_fft, a_fft[..., i].imag, label='Sine Coefficients (Imaginary)')  # Imag signals FR
    axes[2].plot(phase_angles)  # Complex number phase angles
    axes[3].plot(t, signal_ifft)  # Inverse FFT
    
    mp.visualize.IncreaseLegendLinewidth(axes[1].legend(loc=4))
    fig.tight_layout()
    plt.show()

def fft2d():
    # 2D grid signal
    grid = torch.zeros(128, 128)
    x = torch.linspace(0, 2 * torch.pi, grid.shape[0])
    for j in range(grid.shape[1]):
        grid[:, j] += 10 * torch.sin(2 * pi * 3 * x) + 5 * torch.cos(2 * pi * 10 * x)  # 3Hz imag and 5 Hz real
        grid[j, :] += 10 * torch.sin(2 * pi * 3 * x) + 5 * torch.cos(2 * pi * 10 * x)  # 3Hz imag and 5 Hz real
    grid_x, grid_y = torch.meshgrid([torch.arange(grid.shape[0]), torch.arange(grid.shape[1])])
    
    # Compute FFT
    a_fft2 = torch.fft.fft2(grid)  # equivalent to fft(fft(testx,dim=0),dim=1)
    a_fft2 *= 2 / (a_fft2.shape[0] * a_fft2.shape[1])  # Normalization
    a_rfft2 = torch.fft.rfft2(grid)  # equivalent to fft(fft(testx,dim=0),dim=1)[...,:testx.shape[-1]//2+1]
    a_rfft2 *= 2 / (a_rfft2.shape[0] * a_rfft2.shape[1])  # Normalization
    
    # Linear transform on fft results and IFFT
    grid_ifft = torch.fft.ifft2(a_fft2)
    grid_ifft /= 2 / (a_fft2.shape[0] * a_fft2.shape[1])  # Inverse normalization (Fourier space preserves linearity)
    grid_irfft = torch.fft.irfft2(a_rfft2)
    grid_irfft /= 2 / (a_rfft2.shape[0] * a_rfft2.shape[1])  # Inverse normalization (Fourier space preserves linearity)
    
    # Plot
    mp.visualize.PlotTemplate()
    fig, ax = plt.subplots(1, 4, figsize=(18, 7), subplot_kw={'projection': '3d'})
    ax[0].plot_surface(grid_x, grid_y, grid)
    ax[1].plot_surface(grid_x, grid_y, torch.abs(a_fft2))
    ax[2].plot_surface(grid_x[:, :grid_x.shape[1] // 2 + 1], grid_y[:, :grid_x.shape[1] // 2 + 1], torch.abs(a_rfft2))
    # ax[3].plot_surface(grid_x, grid_y, grid_ifft)
    ax[3].plot_surface(grid_x, grid_y, grid_irfft)
    ax[0].set_title('Spatial Signal\ndim0: 10*3Hz and 5*5Hz\ndim1: 10*3Hz and 5*5Hz', fontsize=12)
    ax[1].set_title('FFT2 result')
    ax[2].set_title('RFFT2 result')
    ax[3].set_title('IFFT2 result')
    fig.tight_layout()
    plt.show()

def fftnd():
    grid = torch.randn(8, 16, 32, 64, 128)
    dim = [-3, -2, -1]  # Dimension to compute FFT. The last dimension is compressed by RFFT
    
    a_fftn = torch.fft.fftn(grid, dim=dim)
    a_rfftn = torch.fft.rfftn(grid, dim=dim)
    
    a_ifftn = torch.fft.ifftn(a_fftn, dim=dim) # Imaginary parts of IFFT results are near zero. If the FFT input signal was real, the IFFT(FFT) signal should be real, analytically.
    a_irfftn = torch.fft.irfftn(a_rfftn, dim=dim) # Only returns purely real signals, NO IMAGINARY output with RFFT
    
    mp.visualize.PlotTemplate()
    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    random_idx = torch.randint(0, 8, size=(3, 3))  # 3 samples, 3 randomized indices
    for i in range(3):
        axes[0, i].matshow(grid[random_idx[i, 0], random_idx[i, 1], random_idx[i, 2], :, :])
        axes[1, i].matshow(a_ifftn[random_idx[i, 0], random_idx[i, 1], random_idx[i, 2], :, :])
    plt.show()
    
    pass

fft1d_vector()
# fft1d_array()
# fft2d()
# fftnd()
