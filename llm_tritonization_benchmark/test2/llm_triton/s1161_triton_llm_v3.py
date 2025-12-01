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
    
    # Condition: c[i] < 0
    condition = c_vals < 0.0
    
    # Branch 1: a[i] = c[i] + d[i] * e[i] (when c[i] >= 0)
    a_result = c_vals + d_vals * e_vals
    
    # Branch 2: b[i] = a[i] + d[i] * d[i] (when c[i] < 0)
    b_result = a_vals + d_vals * d_vals
    
    # Store a[i] when condition is false (c[i] >= 0)
    tl.store(a_ptr + offsets, a_result, mask=mask & ~condition)
    
    # Store b[i] when condition is true (c[i] < 0)
    tl.store(b_ptr + offsets, b_result, mask=mask & condition)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )