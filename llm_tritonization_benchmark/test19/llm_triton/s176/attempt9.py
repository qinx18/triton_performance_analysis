import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Build matrix B where B[i, j] = b[i+m-j-1]
    i_indices = torch.arange(m, device=a.device, dtype=torch.long).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device, dtype=torch.long).unsqueeze(0)  # (1, m)
    
    # b_indices[i, j] = i + m - j - 1
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Build the matrix B
    B = b[b_indices]  # (m, m)
    
    # Compute matrix-vector multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(B, c[:m])
    a[:m] += result