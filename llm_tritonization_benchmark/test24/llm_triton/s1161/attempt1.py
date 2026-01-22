import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load values
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Condition: c[i] < 0
    cond_mask = c_vals < 0.0
    
    # Compute both branches
    # If c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    a_if = c_vals + d_vals * e_vals
    
    # If c[i] < 0: b[i] = a[i] + d[i] * d[i]
    b_else = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # Store a values where condition is false (c[i] >= 0)
    a_result = tl.where(cond_mask, a_vals, a_if)
    tl.store(a_ptr + offsets, a_result, mask=mask)
    
    # Store b values where condition is true (c[i] < 0)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    b_result = tl.where(cond_mask, b_else, b_vals)
    tl.store(b_ptr + offsets, b_result, mask=mask)

def s1161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1161_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)