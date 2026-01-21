import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Use PyTorch conv1d for efficient convolution computation
    # Reshape for conv1d: (batch, channels, length)
    b_input = b[:2*m-1].unsqueeze(0).unsqueeze(0)  # (1, 1, 2*m-1)
    c_kernel = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_kernel, padding=0)
    
    # Extract the relevant portion and add to a
    a[:m] += conv_result[0, 0, :m]