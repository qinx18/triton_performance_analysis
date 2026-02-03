import triton
import triton.language as tl
import torch

@triton.jit
def s273_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute d[i] * e[i]
    de_product = d_vals * e_vals
    
    # a[i] += d[i] * e[i]
    a_vals = a_vals + de_product
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    condition = a_vals < 0.0
    b_vals = tl.where(condition, b_vals + de_product, b_vals)
    
    # c[i] += a[i] * d[i]
    c_vals = tl.load(c_ptr + indices, mask=mask)
    c_vals = c_vals + a_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)
    tl.store(b_ptr + indices, b_vals, mask=mask)
    tl.store(c_ptr + indices, c_vals, mask=mask)

def s273_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s273_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)