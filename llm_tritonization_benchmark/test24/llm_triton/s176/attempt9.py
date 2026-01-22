import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Build matrix B where B[i, j] = b[i+m-j-1]
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # Shape: (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # Shape: (1, m)
    
    # Calculate the indices for b: i+m-j-1
    b_indices = i_indices + m - j_indices - 1  # Shape: (m, m)
    
    # Clamp indices to valid range
    b_indices = torch.clamp(b_indices, 0, N-1)
    
    # Gather values from b using the calculated indices
    B = b[b_indices]  # Shape: (m, m)
    
    # Perform matrix-vector multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(B, c[:m])
    a[:m] += result