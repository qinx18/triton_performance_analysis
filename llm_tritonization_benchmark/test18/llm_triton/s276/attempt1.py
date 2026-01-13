import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, mid, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Conditional computation: if (i+1 < mid)
    cond_mask = (indices + 1) < mid
    
    # Compute both branches
    bc_result = b_vals * c_vals
    bd_result = b_vals * d_vals
    
    # Select based on condition
    result = tl.where(cond_mask, bc_result, bd_result)
    
    # Update a
    new_a = a_vals + result
    
    # Store result
    tl.store(a_ptr + indices, new_a, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        mid, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )