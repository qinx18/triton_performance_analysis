import torch
import triton
import triton.language as tl

@triton.jit
def s274_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # a[i] = c[i] + e[i] * d[i]
    a_vals = c_vals + e_vals * d_vals
    
    # if (a[i] > 0.)
    positive_mask = a_vals > 0.0
    
    # b[i] = a[i] + b[i] (when a[i] > 0)
    new_b_vals = tl.where(positive_mask, a_vals + b_vals, b_vals)
    
    # a[i] = d[i] * e[i] (when a[i] <= 0)
    new_a_vals = tl.where(positive_mask, a_vals, d_vals * e_vals)
    
    # Store results
    tl.store(a_ptr + idx, new_a_vals, mask=mask)
    tl.store(b_ptr + idx, new_b_vals, mask=mask)

def s274_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s274_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )