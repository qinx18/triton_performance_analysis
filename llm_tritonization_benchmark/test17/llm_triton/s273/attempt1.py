import torch
import triton
import triton.language as tl

@triton.jit
def s273_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_val = tl.load(a_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    
    # a[i] += d[i] * e[i]
    de_product = d_val * e_val
    a_val = a_val + de_product
    
    # Store updated a
    tl.store(a_ptr + idx, a_val, mask=mask)
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    b_val = tl.load(b_ptr + idx, mask=mask)
    condition = a_val < 0.0
    b_val = tl.where(condition, b_val + de_product, b_val)
    tl.store(b_ptr + idx, b_val, mask=mask)
    
    # c[i] += a[i] * d[i]
    c_val = tl.load(c_ptr + idx, mask=mask)
    c_val = c_val + a_val * d_val
    tl.store(c_ptr + idx, c_val, mask=mask)

def s273_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s273_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)