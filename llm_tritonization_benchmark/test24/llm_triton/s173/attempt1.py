import triton
import triton.language as tl
import torch

@triton.jit
def s173_kernel(a_ptr, b_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute indices for the computation range [0, n/2)
    i_indices = block_start + offsets
    mask = i_indices < n // 2
    
    # Load a[i] and b[i]
    a_vals = tl.load(a_ptr + i_indices, mask=mask)
    b_vals = tl.load(b_ptr + i_indices, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+k]
    output_indices = i_indices + k
    tl.store(a_ptr + output_indices, result, mask=mask)

def s173_triton(a, b, k):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n // 2, BLOCK_SIZE),)
    
    s173_kernel[grid](a, b, k, n, BLOCK_SIZE=BLOCK_SIZE)