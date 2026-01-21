import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create index tensors for the convolution pattern
    i_indices = torch.arange(m, dtype=torch.long, device=a.device)
    j_indices = torch.arange(m, dtype=torch.long, device=a.device)
    
    # Create meshgrid for b indexing: b[i+m-j-1]
    i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
    b_indices = i_grid + m - j_grid - 1
    
    # Extract the relevant portion of b using advanced indexing
    b_matrix = b[b_indices]  # Shape: (m, m)
    
    # Perform matrix-vector multiplication: a[:m] += b_matrix @ c[:m]
    result = torch.matmul(b_matrix, c[:m])
    a[:m] += result