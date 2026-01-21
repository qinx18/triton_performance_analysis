import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    # This is a convolution pattern: a[i] += b[i+m-j-1] * c[j]
    # Rewrite as matrix-vector multiplication to avoid sequential loops
    
    # Create index matrix for b values
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # Shape: (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # Shape: (1, m)
    
    # Calculate b indices: i + m - j - 1
    b_indices = i_indices + m - j_indices - 1  # Shape: (m, m)
    
    # Clamp indices to valid range
    b_indices = torch.clamp(b_indices, 0, N-1)
    
    # Extract b values using advanced indexing
    B_matrix = b[b_indices]  # Shape: (m, m)
    
    # Compute convolution as matrix-vector multiplication
    result = torch.matmul(B_matrix, c[:m])
    
    # Add result to first m elements of a
    a[:m] += result