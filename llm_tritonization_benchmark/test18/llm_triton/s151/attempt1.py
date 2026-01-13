import triton
import triton.language as tl
import torch

@triton.jit
def s151_kernel(a_ptr, b_ptr, n_elements, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < (n_elements - 1)
    
    # Load a[i+m] and b[i]
    a_shifted_indices = indices + m
    a_shifted_mask = mask & (a_shifted_indices < n_elements)
    b_mask = mask
    
    a_shifted = tl.load(a_ptr + a_shifted_indices, mask=a_shifted_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=b_mask, other=0.0)
    
    # Compute a[i] = a[i+m] + b[i]
    result = a_shifted + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.shape[0]
    m = 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n_elements, m, BLOCK_SIZE)