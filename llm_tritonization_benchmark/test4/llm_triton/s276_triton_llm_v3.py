import torch
import triton
import triton.language as tl

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Create condition mask for i+1 < mid (equivalent to offsets+1 < mid)
    cond_mask = (offsets + 1) < mid
    
    # Compute both branches
    branch1 = b_vals * c_vals  # for i+1 < mid
    branch2 = b_vals * d_vals  # for i+1 >= mid
    
    # Select appropriate values based on condition
    result = tl.where(cond_mask, branch1, branch2)
    
    # Update a
    a_vals = a_vals + result
    
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s276_triton(a, b, c, d):
    n_elements = a.numel()
    mid = n_elements // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](a, b, c, d, n_elements, mid, BLOCK_SIZE)