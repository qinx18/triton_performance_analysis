import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # if c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    # else: b[i] = a[i] + d[i] * d[i]
    condition = c_vals >= 0.0
    
    # Compute both branches
    branch_if = c_vals + d_vals * e_vals
    branch_else = a_vals + d_vals * d_vals
    
    # Store results conditionally
    tl.store(a_ptr + offsets, branch_if, mask=mask & condition)
    tl.store(b_ptr + offsets, branch_else, mask=mask & (~condition))

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )