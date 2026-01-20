import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use PyTorch's conv1d for efficient convolution computation
    # This avoids the timeout issue from sequential loops in Triton
    b_input = b[:2*m-1].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 2*m-1)
    c_kernel = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, m)
    
    # Perform convolution
    conv_result = torch.nn.functional.conv1d(b_input, c_kernel, padding=0)
    
    # Add result to a
    a[:m] += conv_result[0, 0, :m]