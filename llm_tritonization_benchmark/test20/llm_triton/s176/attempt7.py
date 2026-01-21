import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # Create index matrices for the convolution pattern
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    
    # Calculate b indices: i + m - j - 1 for each (i, j) pair
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Gather b values using the indices
    b_matrix = b[b_indices]  # (m, m)
    
    # Compute the matrix-vector multiplication: B @ c
    result = torch.matmul(b_matrix, c[:m])
    
    # Add to a
    a[:m] += result