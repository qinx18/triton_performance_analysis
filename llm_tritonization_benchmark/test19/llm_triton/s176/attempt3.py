import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create index matrices for the convolution pattern
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # Shape: (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # Shape: (1, m)
    
    # Compute b indices: i + m - j - 1
    b_indices = i_indices + m - j_indices - 1  # Shape: (m, m)
    
    # Extract the relevant portion of b using advanced indexing
    b_matrix = b[b_indices]  # Shape: (m, m)
    
    # Perform matrix-vector multiplication: sum over j dimension
    result = torch.sum(b_matrix * c[:m].unsqueeze(0), dim=1)
    
    # Add to a
    a[:m] += result