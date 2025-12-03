import torch
import triton
import triton.language as tl

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load a[j] = a[i+1] values
    a_vals = tl.load(a_ptr + indices + 1, mask=mask)
    
    # Compute a[i] = a[j] + b[i]
    result = a_vals + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )