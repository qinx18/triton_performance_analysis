import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute d[i] * e[i] once
    de_product = d_vals * e_vals
    
    # Create condition mask: a[i] > 0
    condition = a_vals > 0.0
    
    # Compute both branches
    # If condition is false: b[i] = -b[i] + d[i] * e[i]
    b_new_false = -b_vals + de_product
    # If condition is true: c[i] = -c[i] + d[i] * e[i]
    c_new_true = -c_vals + de_product
    
    # Apply conditional updates
    b_new = tl.where(condition, b_vals, b_new_false)
    c_new = tl.where(condition, c_new_true, c_vals)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_new + c_new * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, b_new, mask=mask)
    tl.store(c_ptr + indices, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )