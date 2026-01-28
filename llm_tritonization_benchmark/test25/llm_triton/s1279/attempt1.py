import triton
import triton.language as tl
import torch

@triton.jit
def s1279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute conditions
    cond1 = a_vals < 0.0
    cond2 = b_vals > a_vals
    combined_cond = cond1 & cond2
    
    # Update c where conditions are met
    update = d_vals * e_vals
    c_new = tl.where(combined_cond, c_vals + update, c_vals)
    
    # Store result
    tl.store(c_ptr + indices, c_new, mask=mask)

def s1279_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1279_kernel[grid](
        a, b, c, d, e,
        N, BLOCK_SIZE
    )