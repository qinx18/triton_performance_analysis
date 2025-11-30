import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Condition: c[i] < 0.0
    condition = c_vals < 0.0
    
    # Path 1: c[i] >= 0.0 -> a[i] = c[i] + d[i] * e[i]
    a_new = c_vals + d_vals * e_vals
    
    # Path 2: c[i] < 0.0 -> b[i] = a[i] + d[i] * d[i]
    b_new = a_vals + d_vals * d_vals
    
    # Store a values for path 1 (when condition is False)
    tl.store(a_ptr + offsets, a_new, mask=mask & ~condition)
    
    # Store b values for path 2 (when condition is True)
    tl.store(b_ptr + offsets, b_new, mask=mask & condition)

def s1161_triton(a, b, c, d, e):
    n_elements = c.shape[0] - 1  # LEN_1D-1 as per the loop bound
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )