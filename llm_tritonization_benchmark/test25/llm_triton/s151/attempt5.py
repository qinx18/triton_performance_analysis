import triton
import triton.language as tl
import torch

@triton.jit
def s151_kernel(a_ptr, b_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < (n - 1)
    
    # Load a[i+m] and b[i]
    a_shifted_indices = indices + m
    a_shifted_mask = mask & (a_shifted_indices < n)
    a_vals = tl.load(a_ptr + a_shifted_indices, mask=a_shifted_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[i+m] + b[i]
    result = a_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s151_triton(a, b):
    n = a.shape[0]
    m = 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n, m, BLOCK_SIZE=BLOCK_SIZE)