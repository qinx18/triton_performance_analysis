import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create index matrices for the convolution pattern
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    
    # Calculate b indices: i + m - j - 1
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Extract the relevant b values using advanced indexing
    b_matrix = b[b_indices]  # (m, m)
    
    # Perform matrix-vector multiplication: sum over j dimension
    # b_matrix[i, j] * c[j] summed over j
    result = torch.sum(b_matrix * c[:m].unsqueeze(0), dim=1)
    
    # Add to a
    a[:m] += result