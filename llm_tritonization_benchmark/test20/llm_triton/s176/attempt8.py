import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Build convolution matrix B where B[i,j] = b[i+m-j-1]
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Create convolution matrix
    B = b[b_indices]  # (m, m)
    
    # Perform matrix-vector multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(B, c[:m])
    a[:m] += result