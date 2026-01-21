import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Build correlation matrix B where B[i, j] = b[i+m-j-1]
    B = torch.zeros(m, m, dtype=a.dtype, device=a.device)
    
    # Fill the matrix with proper indexing
    for j in range(m):
        for i in range(m):
            b_idx = i + m - j - 1
            if 0 <= b_idx < n:
                B[i, j] = b[b_idx]
    
    # Perform matrix-vector multiplication: a[:m] += B @ c[:m]
    result = B @ c[:m]
    a[:m] += result