import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # Reshape for conv1d: (batch_size, channels, length)
    b_input = b[:2*m].unsqueeze(0).unsqueeze(0)  # (1, 1, 2*m)
    c_weight = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_weight, padding=0)
    
    # Add result to a
    a[:m] += conv_result[0, 0, :m]