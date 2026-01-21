import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create index tensor for b access pattern: b[i+m-j-1]
    # For each i (row) and j (col), compute the b index
    i_indices = torch.arange(m, device=a.device).view(m, 1)  # Shape: (m, 1)
    j_indices = torch.arange(m, device=a.device).view(1, m)  # Shape: (1, m)
    
    # b_indices[i, j] = i + m - j - 1
    b_indices = i_indices + m - j_indices - 1  # Shape: (m, m)
    
    # Create the matrix B where B[i, j] = b[i+m-j-1]
    B = b[b_indices]  # Shape: (m, m)
    
    # Perform matrix-vector multiplication: B @ c[:m]
    # This computes sum over j of B[i, j] * c[j] for each i
    result = torch.matmul(B, c[:m])
    
    # Add result to a[:m]
    a[:m] += result