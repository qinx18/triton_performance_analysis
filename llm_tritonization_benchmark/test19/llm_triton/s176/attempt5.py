import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Build convolution matrix B where B[i, j] = b[i+m-j-1]
    # Create index grids
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    
    # Compute the indices for b: i + m - j - 1
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Extract values from b to form the convolution matrix
    B = b[b_indices]  # (m, m)
    
    # Perform matrix-vector multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(B, c[:m])
    a[:m] += result