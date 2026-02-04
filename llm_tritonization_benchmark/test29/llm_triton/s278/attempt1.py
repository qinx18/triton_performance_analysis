import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute condition: a[i] > 0
    condition = a_vals > 0.0
    
    # Compute both branches
    de_product = d_vals * e_vals
    b_new = -b_vals + de_product
    c_new = -c_vals + de_product
    
    # Select based on condition
    # if a[i] > 0: use c_new, otherwise use b_new
    b_result = tl.where(condition, b_vals, b_new)
    c_result = tl.where(condition, c_new, c_vals)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_result = b_result + c_result * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_result, mask=mask)
    tl.store(b_ptr + indices, b_result, mask=mask)
    tl.store(c_ptr + indices, c_result, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )