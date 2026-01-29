import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # Reshape for conv1d: input (batch, channels, length), weight (out_ch, in_ch, kernel)
    b_input = b.unsqueeze(0).unsqueeze(0)  # (1, 1, n)
    c_weight = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_weight, padding=0)
    
    # Extract the relevant portion and add to a
    start_idx = m - 1
    end_idx = start_idx + m
    a[:m] += conv_result[0, 0, start_idx:end_idx]