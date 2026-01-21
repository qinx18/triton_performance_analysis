import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Build correlation matrix B where B[i, j] = b[i+m-j-1]
    B = torch.zeros(m, m, dtype=a.dtype, device=a.device)
    
    # Fill the matrix efficiently using broadcasting
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Clamp indices to valid range
    b_indices = torch.clamp(b_indices, 0, n-1)
    B = b[b_indices]
    
    # Perform matrix-vector multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(B, c[:m])
    a[:m] += result