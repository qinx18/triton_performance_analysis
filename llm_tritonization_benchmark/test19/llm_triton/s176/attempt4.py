import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create index tensors for the convolution pattern
    i_indices = torch.arange(m, device=a.device, dtype=torch.long).unsqueeze(1)  # Shape: (m, 1)
    j_indices = torch.arange(m, device=a.device, dtype=torch.long).unsqueeze(0)  # Shape: (1, m)
    
    # Compute b indices: i + m - j - 1 for each (i, j) pair
    b_indices = i_indices + m - j_indices - 1  # Shape: (m, m)
    
    # Extract the convolution matrix from b
    b_matrix = b[b_indices]  # Shape: (m, m)
    
    # Perform the convolution: for each i, sum over all j
    # a[i] += sum_j(b[i+m-j-1] * c[j])
    # This is equivalent to: a[:m] += b_matrix @ c[:m]
    result = torch.matmul(b_matrix, c[:m])
    
    # Add to a
    a[:m] += result