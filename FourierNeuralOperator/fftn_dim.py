import torch
import matplotlib.pyplot as plt
import MyPackage as mp
from torch import sin, cos, pi
import numpy as np

def fft1():
    # Set
    Fs = 2000  # Sampling frequency
    T = 1 / Fs  # Sampling time interval
    t_end = 0.5  # End time
    t = torch.arange(0, t_end, T)  # Time vector
    signal = 0.6 * sin(2 * pi * 60 * t) + sin(2 * pi * 120 * t)  # 60Hz(-pi/2 shifted) and 120Hz(non-shifted) cosine sum
    # signal = cos(2*pi*t)  # 60Hz(-pi/2 shifted) and 120Hz(non-shifted) cosine sum
    # signal += 0.1 * torch.randn(t.shape[0], )  # Small Gaussian noise
    n_samples = signal.shape[0]
    
    # Compute FFT
    f_fft = torch.arange(n_samples) * (Fs / n_samples)  # Frequency
    f_fft = f_fft[:n_samples // 2 + 1]  # Single sided FFT
    a_fft = torch.fft.fft(signal) / n_samples  # Normalize with factor of len(signal)
    a_fft = a_fft[:n_samples // 2 + 1]  # Single sided FFT
    a_fft *= 2  # Single sided FFT
    phase_angles = torch.angle(a_fft) * 180 / pi
    
    # IFFT
    signal_ifft = torch.fft.irfft(a_fft) / 2 * n_samples  # Use iRfft since single sided.
    
    # Plot
    mp.visualize.PlotTemplate()
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    axes[0].plot(t, signal)
    axes[1].plot(f_fft, a_fft.real, label='Cosine Coefficients (Real)')
    axes[1].plot(f_fft, a_fft.imag, label='Sine Coefficients (Imaginary)')
    axes[2].plot(phase_angles)
    axes[3].plot(t, signal_ifft)
    
    mp.visualize.IncreaseLegendLinewidth(axes[1].legend(loc=4))
    fig.tight_layout()
    plt.show()

def fft2():
    testx = torch.zeros(64, 64)
    x = torch.linspace(0, 2 * torch.pi, testx.shape[0])
    for j in range(testx.shape[1]):
        testx[:, j] += 10 * torch.sin(1 * x) + 20 * torch.cos(5 * x)
        testx[j, :] += 10 * torch.sin(1 * x) + 20 * torch.cos(5 * x)
    
    mp.visualize.PlotTemplate()
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].matshow(testx)
    ax[1].matshow(torch.fft.fft2(testx).real)
    ax[2].matshow(torch.fft.rfft2(testx).real)
    fig.tight_layout()
    plt.show()


fft1()
