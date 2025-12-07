import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_val = tl.load(a_ptr + idx, mask=mask)
    b_val = tl.load(b_ptr + idx, mask=mask)
    c_val = tl.load(c_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    
    # Condition: a[i] > 0
    condition = a_val > 0.0
    
    # Compute both branches
    de_product = d_val * e_val
    b_new = -b_val + de_product  # Branch when a[i] <= 0
    c_new = -c_val + de_product  # Branch when a[i] > 0
    
    # Select which branch to use for b and c updates
    b_result = tl.where(condition, b_val, b_new)  # Keep b unchanged if condition true
    c_result = tl.where(condition, c_new, c_val)  # Keep c unchanged if condition false
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_result = b_result + c_result * d_val
    
    # Store results
    tl.store(a_ptr + idx, a_result, mask=mask)
    tl.store(b_ptr + idx, b_result, mask=mask)
    tl.store(c_ptr + idx, c_result, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )