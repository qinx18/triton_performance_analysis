import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Create condition mask
    cond_mask = c_vals < 0.0
    
    # For c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    if_result = c_vals + d_vals * e_vals
    
    # For c[i] < 0: b[i] = a[i] + d[i] * d[i]
    else_result = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # Update a when condition is false (c[i] >= 0)
    if_store_mask = mask & (~cond_mask)
    tl.store(a_ptr + offsets, if_result, mask=if_store_mask)
    
    # Update b when condition is true (c[i] < 0)
    else_store_mask = mask & cond_mask
    tl.store(b_ptr + offsets, else_result, mask=else_store_mask)

def s1161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1161_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)