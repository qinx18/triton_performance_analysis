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
    a = tl.load(a_ptr + indices, mask=mask)
    b = tl.load(b_ptr + indices, mask=mask)
    c = tl.load(c_ptr + indices, mask=mask)
    d = tl.load(d_ptr + indices, mask=mask)
    e = tl.load(e_ptr + indices, mask=mask)
    
    # Condition: a[i] > 0
    condition = a > 0.0
    
    # Compute both branches
    de = d * e
    b_new = -b + de  # For a[i] <= 0
    c_new = -c + de  # For a[i] > 0
    
    # Select based on condition
    b_result = tl.where(condition, b, b_new)
    c_result = tl.where(condition, c_new, c)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_result + c_result * d
    
    # Store results
    tl.store(a_ptr + indices, a_new, mask=mask)
    tl.store(b_ptr + indices, b_result, mask=mask)
    tl.store(c_ptr + indices, c_result, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )