import torch
import triton
import triton.language as tl

@triton.jit
def s273_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_val = tl.load(a_ptr + offsets, mask=mask)
    d_val = tl.load(d_ptr + offsets, mask=mask)
    e_val = tl.load(e_ptr + offsets, mask=mask)
    b_val = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute d[i] * e[i] once
    de_product = d_val * e_val
    
    # a[i] += d[i] * e[i]
    a_new = a_val + de_product
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    condition = a_new < 0.0
    b_new = tl.where(condition, b_val + de_product, b_val)
    
    # c[i] += a[i] * d[i]
    c_val = tl.load(c_ptr + offsets, mask=mask)
    c_new = c_val + a_new * d_val
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(c_ptr + offsets, c_new, mask=mask)

def s273_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s273_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)