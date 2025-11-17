"""
1D Fast Fourier Transform (FFT) - Baseline Implementation

FFT is a fundamental signal processing operation that computes the Discrete Fourier Transform
efficiently. It's widely used in audio processing, image processing, and scientific computing.

Baseline approach: Using PyTorch's torch.fft.fft which calls cuFFT (NVIDIA's optimized FFT library)

Time Complexity: O(N log N)
Space Complexity: O(N)

Use cases:
- Signal processing and filtering
- Convolution operations
- Spectral analysis
- Scientific simulations
"""

import torch


def fft_baseline(x):
    """
    Compute 1D FFT using PyTorch (cuFFT backend)

    Args:
        x: Complex input tensor of shape (batch_size, n)
           where n is the sequence length (must be power of 2 for optimal performance)

    Returns:
        Complex output tensor of same shape containing frequency domain representation

    Note:
        PyTorch's torch.fft.fft uses NVIDIA's cuFFT library, which is highly optimized
        and uses mixed-radix algorithms, vendor-specific optimizations, and careful
        memory management.
    """
    return torch.fft.fft(x, dim=-1)


def fft_baseline_real(x):
    """
    Compute 1D FFT for real-valued input

    Args:
        x: Real input tensor of shape (batch_size, n)

    Returns:
        Complex output tensor of shape (batch_size, n) in frequency domain

    Note:
        For real inputs, torch.fft.rfft is more efficient as it only computes
        n//2+1 unique coefficients (due to conjugate symmetry), but for fair
        comparison we use full fft.
    """
    return torch.fft.fft(x, dim=-1)


def ifft_baseline(x):
    """
    Compute inverse 1D FFT using PyTorch (cuFFT backend)

    Args:
        x: Complex input tensor in frequency domain

    Returns:
        Complex output tensor in time domain
    """
    return torch.fft.ifft(x, dim=-1)


# Alternative: Manual DFT for verification (extremely slow, only for testing)
def naive_dft(x):
    """
    Naive O(N^2) DFT implementation for correctness verification

    DO NOT USE FOR BENCHMARKING - This is extremely slow!
    Only use for verifying correctness on small inputs.

    Args:
        x: Complex tensor of shape (batch_size, n)

    Returns:
        Complex tensor of same shape
    """
    import math

    batch_size, n = x.shape
    result = torch.zeros_like(x)

    for b in range(batch_size):
        for k in range(n):
            sum_val = 0.0 + 0.0j
            for t in range(n):
                angle = -2 * math.pi * k * t / n
                twiddle = complex(math.cos(angle), math.sin(angle))
                sum_val += x[b, t].item() * twiddle
            result[b, k] = sum_val

    return result
