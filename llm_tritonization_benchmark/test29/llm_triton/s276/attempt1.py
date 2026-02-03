import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Conditional computation: if (i+1 < mid)
    cond_mask = (offsets + 1) < mid
    
    # Compute both branches
    branch1 = a_vals + b_vals * c_vals  # i+1 < mid
    branch2 = a_vals + b_vals * d_vals  # else
    
    # Select based on condition
    result = tl.where(cond_mask, branch1, branch2)
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a