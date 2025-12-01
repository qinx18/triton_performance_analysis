import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets * inc
    
    mask = indices < n_elements
    
    # Load a[i + inc] and b[i]
    a_inc_indices = indices + inc
    a_inc_mask = mask & (a_inc_indices < (n_elements + inc))
    
    a_inc_vals = tl.load(a_ptr + a_inc_indices, mask=a_inc_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[i + inc] + b[i]
    result = a_inc_vals + b_vals
    
    # Store result to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(triton.cdiv(n_elements, inc), BLOCK_SIZE)
    
    s175_kernel[grid_size,](
        a, b, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )