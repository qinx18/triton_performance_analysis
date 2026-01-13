import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n
    
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Check condition: c[i] < 0
    cond_mask = c_vals < 0.0
    
    # If c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    if_result = c_vals + d_vals * e_vals
    
    # If c[i] < 0: b[i] = a[i] + d[i] * d[i]
    else_result = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # For if branch (c[i] >= 0)
    if_store_mask = mask & (cond_mask == False)
    if tl.any(if_store_mask):
        tl.store(a_ptr + idx, if_result, mask=if_store_mask)
    
    # For else branch (c[i] < 0)
    else_store_mask = mask & cond_mask
    if tl.any(else_store_mask):
        tl.store(b_ptr + idx, else_result, mask=else_store_mask)

def s1161_triton(a, b, c, d, e):
    n = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )