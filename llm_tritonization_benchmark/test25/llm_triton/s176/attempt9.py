import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch conv1d for efficient convolution computation
    # Reshape for conv1d: (batch, channels, length)
    b_input = b.unsqueeze(0).unsqueeze(0)  # (1, 1, n)
    c_kernel = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Compute convolution with appropriate padding
    conv_result = torch.nn.functional.conv1d(b_input, c_kernel, padding=0)
    
    # Extract the relevant portion and add to a
    result_len = conv_result.shape[2]
    start_idx = result_len - m
    a[:m] += conv_result[0, 0, start_idx:]