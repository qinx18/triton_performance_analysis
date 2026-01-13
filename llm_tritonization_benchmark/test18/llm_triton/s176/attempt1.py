import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # This is a convolution pattern: a[i] += sum_j(b[i+m-j-1] * c[j])
    # Rewrite as matrix-vector multiplication to avoid sequential loops
    
    # Build the Toeplitz-like matrix B where B[i, j] = b[i+m-j-1]
    # Using torch operations for efficiency
    B = torch.zeros(m, m, dtype=a.dtype, device=a.device)
    
    for j in range(m):
        start_idx = m - j - 1
        end_idx = start_idx + m
        B[:, j] = b[start_idx:end_idx]
    
    # Compute the matrix-vector product
    result = torch.matmul(B, c[:m])
    
    # Add to accumulator
    a[:m] += result