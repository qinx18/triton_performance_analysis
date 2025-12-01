import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Condition: c[i] < 0
    condition = c_vals < 0.0
    
    # Branch 1: c[i] >= 0 -> a[i] = c[i] + d[i] * e[i]
    a_result = c_vals + d_vals * e_vals
    
    # Branch 2: c[i] < 0 -> b[i] = a[i] + d[i] * d[i]
    b_result = a_vals + d_vals * d_vals
    
    # Store a values when condition is false (c[i] >= 0)
    a_store_mask = mask & ~condition
    tl.store(a_ptr + offsets, a_result, mask=a_store_mask)
    
    # Store b values when condition is true (c[i] < 0)
    b_store_mask = mask & condition
    tl.store(b_ptr + offsets, b_result, mask=b_store_mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )