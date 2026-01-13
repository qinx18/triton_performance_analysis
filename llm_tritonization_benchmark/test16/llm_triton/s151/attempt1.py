import torch
import triton
import triton.language as tl

@triton.jit
def s151_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Compute block of indices to process
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices (need to process N-1 elements)
    mask = indices < (N - 1)
    
    # Load a[i+1] and b[i] for valid indices
    a_shifted = tl.load(a_ptr + indices + 1, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_shifted + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s151_triton(a, b):
    N = a.shape[0]
    
    # Only process N-1 elements
    n_elements = N - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s151_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )