import torch
import triton
import triton.language as tl

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Conditional computation: if (i+1 < mid)
    condition = (idx + 1) < mid
    
    # Compute both branches
    branch1 = a_vals + b_vals * c_vals  # i+1 < mid
    branch2 = a_vals + b_vals * d_vals  # else
    
    # Select based on condition
    result = tl.where(condition, branch1, branch2)
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE
    )