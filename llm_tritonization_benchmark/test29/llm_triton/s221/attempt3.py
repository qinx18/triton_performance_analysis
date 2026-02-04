import triton
import triton.language as tl
import torch

def s221_triton(a, b, c, d):
    n = a.shape[0]
    
    # First statement: a[i] += c[i] * d[i] (can be parallelized)
    a[1:] += c[1:] * d[1:]
    
    # Second statement: b[i] = b[i-1] + a[i] + d[i] (sequential dependency)
    # This is a prefix sum pattern that must be computed sequentially
    for i in range(1, n):
        b[i] = b[i-1] + a[i] + d[i]
    
    return a, b