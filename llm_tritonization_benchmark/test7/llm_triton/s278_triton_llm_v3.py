import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute condition: a[i] > 0
    condition = a_vals > 0.0
    
    # Compute both paths
    de_product = d_vals * e_vals
    b_new = -b_vals + de_product  # Path when a[i] <= 0
    c_new = -c_vals + de_product  # Path when a[i] > 0
    
    # Select which path to use for b and c updates
    b_final = tl.where(condition, b_vals, b_new)  # Keep old b if a > 0, else update
    c_final = tl.where(condition, c_new, c_vals)  # Update c if a > 0, else keep old
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, b_final, mask=mask)
    tl.store(c_ptr + indices, c_final, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )