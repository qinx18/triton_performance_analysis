import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load data
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Condition: c[i] < 0.0
    condition = c_vals < 0.0
    
    # If c[i] >= 0.0: a[i] = c[i] + d[i] * e[i]
    a_new = c_vals + d_vals * e_vals
    a_result = tl.where(condition, a_vals, a_new)
    
    # If c[i] < 0.0: b[i] = a[i] + d[i] * d[i]
    b_new = a_vals + d_vals * d_vals
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    b_result = tl.where(condition, b_new, b_vals)
    
    # Store results
    tl.store(a_ptr + offsets, a_result, mask=mask)
    tl.store(b_ptr + offsets, b_result, mask=mask)

def s1161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    if n <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1161_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE)