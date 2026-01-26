import torch
import triton
import triton.language as tl

def s176_triton(a, b, c, m):
    # This is a convolution pattern: a[i] += sum over j of b[i+m-j-1] * c[j]
    # We need to be careful with the indexing to match the C code exactly
    
    # Create index matrix for vectorized computation
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    
    # Compute b indices: i + m - j - 1
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Extract the relevant portions
    b_matrix = b[b_indices]  # (m, m)
    c_vector = c[:m].unsqueeze(0)  # (1, m)
    
    # Compute the convolution: for each i, sum over j of b[i+m-j-1] * c[j]
    result = torch.sum(b_matrix * c_vector, dim=1)  # (m,)
    
    # Add to output
    a[:m] += result