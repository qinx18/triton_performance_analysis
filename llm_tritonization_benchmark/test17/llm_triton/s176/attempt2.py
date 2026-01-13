import torch
import triton
import triton.language as tl

def s176_triton(a, b, c, m):
    # Use efficient matrix multiplication approach for the convolution
    n = a.shape[0]
    
    # Build the convolution matrix B where B[i, j] = b[i+m-j-1]
    B = torch.zeros(m, m, dtype=a.dtype, device=a.device)
    
    for j in range(m):
        for i in range(m):
            b_idx = i + m - j - 1
            if 0 <= b_idx < n:
                B[i, j] = b[b_idx]
    
    # Compute the convolution as matrix-vector multiplication
    result = B @ c[:m]
    a[:m] += result