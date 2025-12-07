import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Branch condition
    cond = b_vals < 0.0
    
    # Path 1: a[i] = c[i] + d[i] * e[i]
    a_result = c_vals + d_vals * e_vals
    
    # Path 2: c[i+1] = a[i] + d[i] * d[i]
    c_result = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # For a[i] when b[i] >= 0
    tl.store(a_ptr + idx, a_result, mask=mask & ~cond)
    
    # For c[i+1] when b[i] < 0
    store_mask = mask & cond & (idx + 1 < (n_elements + 1))
    tl.store(c_ptr + idx + 1, c_result, mask=store_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s161_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)