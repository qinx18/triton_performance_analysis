import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # Reshape for conv1d: (batch_size, channels, length)
    b_padded = torch.nn.functional.pad(b, (0, m-1))  # Pad b to handle boundary
    b_reshaped = b_padded.unsqueeze(0).unsqueeze(0)  # (1, 1, n+m-1)
    
    # Flip c and reshape for convolution kernel
    c_flipped = torch.flip(c[:m], [0]).unsqueeze(0).unsqueeze(0)  # (1, 1, m)
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_reshaped, c_flipped, padding=0)
    
    # Extract the relevant portion and add to a
    a[:m] += conv_result[0, 0, m-1:2*m-1]