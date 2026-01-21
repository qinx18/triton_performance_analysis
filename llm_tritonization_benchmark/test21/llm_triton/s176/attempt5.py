import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Create index matrices for the convolution pattern
    i_indices = torch.arange(m, device=a.device, dtype=torch.long).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device, dtype=torch.long).unsqueeze(0)  # (1, m)
    
    # Compute b indices: i + m - j - 1
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Extract the matrix of b values
    B = b[b_indices]  # (m, m)
    
    # Perform matrix-vector multiplication: a[:m] += B @ c[:m]
    result = torch.matmul(B, c[:m])
    a[:m] += result