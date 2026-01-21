import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Build matrix B where B[i, j] = b[i+m-j-1]
    i_indices = torch.arange(m, dtype=torch.long, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, dtype=torch.long, device=a.device).unsqueeze(0)  # (1, m)
    
    # Calculate b indices: i + m - j - 1
    # For i=0, j=0: index = 0 + m - 0 - 1 = m - 1
    # For i=m-1, j=m-1: index = m-1 + m - (m-1) - 1 = m - 1
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Extract matrix from b
    b_matrix = b[b_indices]  # Shape: (m, m)
    
    # Perform matrix-vector multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(b_matrix, c[:m])
    a[:m] += result