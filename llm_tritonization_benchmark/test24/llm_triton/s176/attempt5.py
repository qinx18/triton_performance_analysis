import torch
import triton
import triton.language as tl

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    # Create index matrices for the convolution pattern
    # a[i] += b[i+m-j-1] * c[j]
    # This is equivalent to: a[:m] += sum over j of (b[m-j-1:2*m-j-1] * c[j])
    
    # Build the computation using matrix operations
    # For each i in [0, m), we need sum over j of b[i+m-j-1] * c[j]
    
    for j in range(m):
        # For this j, compute a[i] += b[i+m-j-1] * c[j] for all i
        start_idx = m - j - 1
        a[:m] += b[start_idx:start_idx + m] * c[j]