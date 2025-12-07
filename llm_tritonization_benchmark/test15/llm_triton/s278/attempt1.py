import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)
    c = tl.load(c_ptr + idx, mask=mask)
    d = tl.load(d_ptr + idx, mask=mask)
    e = tl.load(e_ptr + idx, mask=mask)
    
    # Compute d * e once
    de = d * e
    
    # Conditional computation
    condition = a > 0.0
    
    # If a[i] <= 0: b[i] = -b[i] + d[i] * e[i]
    b_new = tl.where(condition, b, -b + de)
    
    # If a[i] > 0: c[i] = -c[i] + d[i] * e[i]
    c_new = tl.where(condition, -c + de, c)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_new + c_new * d
    
    # Store results
    tl.store(a_ptr + idx, a_new, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)
    tl.store(c_ptr + idx, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)