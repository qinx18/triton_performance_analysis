import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Build the convolution matrix explicitly
    # For each j, we need b[i+m-j-1] * c[j] added to a[i]
    # This can be computed as a matrix multiplication
    
    # Create index matrix for b access pattern
    i_indices = torch.arange(m, device=a.device).unsqueeze(1)  # (m, 1)
    j_indices = torch.arange(m, device=a.device).unsqueeze(0)  # (1, m)
    b_indices = i_indices + m - j_indices - 1  # (m, m)
    
    # Extract the relevant portion of b using advanced indexing
    B = b[b_indices]  # (m, m) matrix where B[i,j] = b[i+m-j-1]
    
    # Perform matrix-vector multiplication: a[:m] += B @ c[:m]
    a[:m] += torch.matmul(B, c[:m])