import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Check condition: c[i] < 0
    condition = c_vals < 0.0
    
    # Path 1: c[i] >= 0, compute a[i] = c[i] + d[i] * e[i]
    new_a = c_vals + d_vals * e_vals
    
    # Path 2: c[i] < 0, compute b[i] = a[i] + d[i] * d[i]
    new_b = a_vals + d_vals * d_vals
    
    # Store results conditionally
    # For a: only store when c[i] >= 0
    a_mask = mask & (~condition)
    tl.store(a_ptr + offsets, new_a, mask=a_mask)
    
    # For b: only store when c[i] < 0
    b_mask = mask & condition
    tl.store(b_ptr + offsets, new_b, mask=b_mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )