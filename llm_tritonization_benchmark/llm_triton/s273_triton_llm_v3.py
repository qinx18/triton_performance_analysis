import triton
import triton.language as tl
import torch

@triton.jit
def s273_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # a[i] += d[i] * e[i]
    de_product = d_vals * e_vals
    new_a_vals = a_vals + de_product
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    condition = new_a_vals < 0.0
    new_b_vals = tl.where(condition, b_vals + de_product, b_vals)
    
    # c[i] += a[i] * d[i]
    new_c_vals = c_vals + new_a_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    tl.store(b_ptr + offsets, new_b_vals, mask=mask)
    tl.store(c_ptr + offsets, new_c_vals, mask=mask)

def s273_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s273_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )