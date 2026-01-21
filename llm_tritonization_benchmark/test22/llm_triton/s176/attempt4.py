import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create index tensors for the convolution pattern
    i_indices = torch.arange(m, device=a.device).view(m, 1)
    j_indices = torch.arange(m, device=a.device).view(1, m)
    
    # For each (i,j) pair, compute the b index: i + m - j - 1
    b_indices = i_indices + m - j_indices - 1
    
    # Build matrix B where B[i,j] = b[i+m-j-1]
    B = b[b_indices]
    
    # Compute matrix-vector product: B @ c[:m]
    # This is equivalent to: sum over j of B[i,j] * c[j] for each i
    result = torch.matmul(B, c[:m])
    
    # Add to a[:m]
    a[:m] += result