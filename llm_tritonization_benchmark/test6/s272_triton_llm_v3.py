import torch
import triton
import triton.language as tl

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    e_vals = tl.load(e_ptr + idx, mask=mask)
    condition = e_vals >= t
    
    # Only load other values where condition is true
    combined_mask = mask & condition
    
    c_vals = tl.load(c_ptr + idx, mask=combined_mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=combined_mask, other=0.0)
    
    # Update a[i] += c[i] * d[i]
    a_vals = tl.load(a_ptr + idx, mask=combined_mask, other=0.0)
    a_new = a_vals + c_vals * d_vals
    tl.store(a_ptr + idx, a_new, mask=combined_mask)
    
    # Update b[i] += c[i] * c[i]
    b_vals = tl.load(b_ptr + idx, mask=combined_mask, other=0.0)
    b_new = b_vals + c_vals * c_vals
    tl.store(b_ptr + idx, b_new, mask=combined_mask)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e, t, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )