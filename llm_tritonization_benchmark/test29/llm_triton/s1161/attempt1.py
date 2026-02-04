import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Branch condition: c[i] < 0
    cond_mask = c_vals < 0.0
    
    # Else branch: a[i] = c[i] + d[i] * e[i] (when c[i] >= 0)
    else_mask = mask & (~cond_mask)
    a_result = c_vals + d_vals * e_vals
    tl.store(a_ptr + idx, a_result, mask=else_mask)
    
    # If branch: b[i] = a[i] + d[i] * d[i] (when c[i] < 0)
    # Need to reload a[i] in case it was updated by else branch
    a_vals_updated = tl.load(a_ptr + idx, mask=mask)
    if_mask = mask & cond_mask
    b_result = a_vals_updated + d_vals * d_vals
    tl.store(b_ptr + idx, b_result, mask=if_mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )