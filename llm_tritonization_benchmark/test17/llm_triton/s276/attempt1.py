import torch
import triton
import triton.language as tl

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, mid, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute condition: i+1 < mid (indices are 0-based, so indices+1 < mid)
    cond = (indices + 1) < mid
    
    # Select multiplication factor based on condition
    mult_vals = tl.where(cond, c_vals, d_vals)
    
    # Compute result
    result = a_vals + b_vals * mult_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d, mid, n_elements, BLOCK_SIZE
    )