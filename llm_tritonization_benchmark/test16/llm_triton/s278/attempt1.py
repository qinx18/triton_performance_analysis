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
    
    # Conditional logic: if a[i] > 0
    condition = a > 0.0
    
    # Compute both branches
    new_b = -b + d * e
    new_c = -c + d * e
    
    # Update b and c based on condition
    # If a[i] <= 0: update b[i], keep c[i] unchanged
    # If a[i] > 0: update c[i], keep b[i] unchanged  
    b = tl.where(condition, b, new_b)
    c = tl.where(condition, new_c, c)
    
    # Always compute a[i] = b[i] + c[i] * d[i]
    new_a = b + c * d
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, b, mask=mask)
    tl.store(c_ptr + idx, c, mask=mask)

def s278_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e, N, BLOCK_SIZE
    )