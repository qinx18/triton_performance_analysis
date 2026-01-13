import torch
import triton
import triton.language as tl

def s176_triton(a, b, c, m):
    # This is a convolution pattern that needs to be computed efficiently
    # The original nested loop: a[i] += b[i+m-j-1] * c[j]
    # This creates a matrix-vector multiplication pattern
    
    # Build the computation matrix B where B[i,j] = b[i+m-j-1]
    # Then compute a[:m] += B @ c[:m]
    
    # Create index tensors for the matrix construction
    i_indices = torch.arange(m, device=a.device, dtype=torch.long)
    j_indices = torch.arange(m, device=a.device, dtype=torch.long)
    
    # Create the access pattern: i+m-j-1 for each (i,j) pair
    i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
    b_indices = i_grid + m - j_grid - 1
    
    # Extract the matrix B from array b using the computed indices
    B = b[b_indices]  # Shape: (m, m)
    
    # Compute the matrix-vector multiplication
    result = torch.matmul(B, c[:m])
    
    # Add to the accumulator array
    a[:m] += result