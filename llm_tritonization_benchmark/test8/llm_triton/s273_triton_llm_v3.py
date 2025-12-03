import torch
import triton
import triton.language as tl

@triton.jit
def s273_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # a[i] += d[i] * e[i]
    de_product = d_vals * e_vals
    a_vals = a_vals + de_product
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    negative_mask = a_vals < 0.0
    b_update = tl.where(negative_mask, de_product, 0.0)
    b_vals = b_vals + b_update
    
    # c[i] += a[i] * d[i]
    c_vals = tl.load(c_ptr + indices, mask=mask)
    c_vals = c_vals + a_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)
    tl.store(b_ptr + indices, b_vals, mask=mask)
    tl.store(c_ptr + indices, c_vals, mask=mask)

def s273_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s273_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )