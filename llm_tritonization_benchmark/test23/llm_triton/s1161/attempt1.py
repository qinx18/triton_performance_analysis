import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Create condition mask: c[i] < 0
    cond_mask = c_vals < 0.0
    
    # For condition false (c[i] >= 0): a[i] = c[i] + d[i] * e[i]
    a_new = c_vals + d_vals * e_vals
    
    # For condition true (c[i] < 0): b[i] = a[i] + d[i] * d[i]
    b_new = a_vals + d_vals * d_vals
    
    # Store a values where condition is false
    a_write_mask = mask & (~cond_mask)
    tl.store(a_ptr + offsets, a_new, mask=a_write_mask)
    
    # Store b values where condition is true
    b_write_mask = mask & cond_mask
    tl.store(b_ptr + offsets, b_new, mask=b_write_mask)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )