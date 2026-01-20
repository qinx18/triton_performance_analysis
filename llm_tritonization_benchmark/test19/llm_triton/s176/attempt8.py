import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create index matrices for the convolution pattern
    # a[i] += b[i+m-j-1] * c[j] for j in [0, m), i in [0, m)
    
    # Create indices
    i_indices = torch.arange(m, device=a.device, dtype=torch.long).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device, dtype=torch.long).unsqueeze(0)  # (1, m)
    
    # Compute b indices: i+m-j-1
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Extract the relevant portions and perform matrix multiplication
    b_matrix = b[b_indices]  # (m, m)
    c_vector = c[:m]  # (m,)
    
    # Compute the convolution as matrix-vector multiplication
    result = torch.matmul(b_matrix, c_vector)
    
    # Add to a
    a[:m] += result