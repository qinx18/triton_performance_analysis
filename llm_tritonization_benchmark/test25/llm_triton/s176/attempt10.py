import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create Toeplitz-like matrix for the convolution pattern
    # a[i] += b[i+m-j-1] * c[j] becomes matrix multiplication
    indices_i = torch.arange(m, device=a.device).unsqueeze(1)  # Shape: (m, 1)
    indices_j = torch.arange(m, device=a.device).unsqueeze(0)  # Shape: (1, m)
    
    # Compute b indices: i + m - j - 1
    b_indices = indices_i + m - indices_j - 1
    
    # Clamp indices to valid range
    b_indices = torch.clamp(b_indices, 0, n - 1)
    
    # Gather b values according to the pattern
    B = b[b_indices]  # Shape: (m, m)
    
    # Perform matrix-vector multiplication
    result = torch.matmul(B, c[:m])
    
    # Add to a
    a[:m] += result