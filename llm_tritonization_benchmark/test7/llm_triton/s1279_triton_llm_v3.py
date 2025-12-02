import torch
import triton
import triton.language as tl

@triton.jit
def s1279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # if (a[i] < 0.) { if (b[i] > a[i]) { c[i] += d[i] * e[i]; } }
    cond1 = a_vals < 0.0
    cond2 = b_vals > a_vals
    update_mask = cond1 & cond2 & mask
    
    increment = d_vals * e_vals
    new_c = tl.where(update_mask, c_vals + increment, c_vals)
    
    tl.store(c_ptr + offsets, new_c, mask=mask)

def s1279_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )