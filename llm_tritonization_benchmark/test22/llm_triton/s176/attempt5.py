import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create shifted matrix B where B[i, j] = b[i+m-j-1]
    i_indices = torch.arange(m, device=a.device).view(-1, 1)
    j_indices = torch.arange(m, device=a.device).view(1, -1)
    
    # Calculate b indices: i+m-j-1
    b_indices = i_indices + m - j_indices - 1
    
    # Build matrix B
    B = b[b_indices]
    
    # Compute convolution as matrix multiplication
    result = torch.matmul(B, c[:m])
    
    # Update a
    a[:m] += result