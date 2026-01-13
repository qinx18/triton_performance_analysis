import triton
import triton.language as tl
import torch

@triton.jit
def s273_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load input values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # a[i] += d[i] * e[i]
    de_product = d_vals * e_vals
    a_vals = a_vals + de_product
    
    # Store updated a values
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    a_negative = a_vals < 0.0
    b_update = tl.where(a_negative, b_vals + de_product, b_vals)
    tl.store(b_ptr + idx, b_update, mask=mask)
    
    # c[i] += a[i] * d[i]
    c_vals = tl.load(c_ptr + idx, mask=mask)
    c_vals = c_vals + a_vals * d_vals
    tl.store(c_ptr + idx, c_vals, mask=mask)

def s273_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s273_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)