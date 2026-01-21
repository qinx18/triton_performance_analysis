import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Build shifted matrix B where B[i, j] = b[i+m-j-1]
    B = torch.zeros(m, m, dtype=a.dtype, device=a.device)
    
    for j in range(m):
        start_idx = m - j - 1
        end_idx = start_idx + m
        B[:, j] = b[start_idx:end_idx]
    
    # Compute a[:m] += B @ c[:m]
    result = B @ c[:m]
    a[:m] += result