import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Use torch convolution for efficient computation
    # Create indices for the shifted access pattern
    B = torch.zeros(m, m, dtype=a.dtype, device=a.device)
    for j in range(m):
        start_idx = m - j - 1
        end_idx = start_idx + m
        B[:, j] = b[start_idx:end_idx]
    
    # Compute matrix-vector multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(B, c[:m])
    a[:m] += result