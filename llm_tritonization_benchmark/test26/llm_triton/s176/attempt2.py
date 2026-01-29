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
    
    # Calculate padding to get the correct output size
    padding = m - 1
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_weight, padding=padding)
    
    # Extract the first m elements and add to a
    a[:m] += conv_result[0, 0, :m]