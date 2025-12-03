import torch
import triton
import triton.language as tl

@triton.jit
def s1279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = block_id * BLOCK_SIZE
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Check if a[i] < 0
    cond1 = a_vals < 0.0
    
    # Check if b[i] > a[i]
    cond2 = b_vals > a_vals
    
    # Combined condition: both conditions must be true
    combined_cond = cond1 & cond2
    
    # Compute d[i] * e[i]
    product = d_vals * e_vals
    
    # Update c[i] only where both conditions are met
    c_vals = tl.where(combined_cond, c_vals + product, c_vals)
    
    # Store results
    tl.store(c_ptr + idx, c_vals, mask=mask)

def s1279_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1279_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )