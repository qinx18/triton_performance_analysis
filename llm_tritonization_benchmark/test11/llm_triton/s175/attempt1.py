import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, n_elements, inc, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices (must be < n_elements-1 and align with stride)
    valid_mask = (indices < n_elements - 1) & ((indices % inc) == 0)
    
    # Load values where mask is valid
    a_read_indices = indices + inc
    read_mask = valid_mask & (a_read_indices < n_elements)
    
    a_vals = tl.load(a_ptr + a_read_indices, mask=read_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=valid_mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=valid_mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, n_elements, inc,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a