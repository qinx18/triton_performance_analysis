import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # Pad b to handle the shifted access pattern
    b_padded = torch.cat([b, torch.zeros(m-1, device=b.device, dtype=b.dtype)])
    
    # Reshape for conv1d: (batch, channels, length)
    b_input = b_padded.unsqueeze(0).unsqueeze(0)
    c_kernel = c[:m].flip(0).unsqueeze(0).unsqueeze(0)
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_kernel, padding=0)
    
    # Extract the relevant portion and add to a
    a[:m] += conv_result[0, 0, m-1:2*m-1]