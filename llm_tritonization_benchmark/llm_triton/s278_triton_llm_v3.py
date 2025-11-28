import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Implement the conditional logic
    condition = a > 0.0
    
    # When a[i] <= 0: b[i] = -b[i] + d[i] * e[i]
    b_new = tl.where(condition, b, -b + d * e)
    
    # When a[i] > 0: c[i] = -c[i] + d[i] * e[i]
    c_new = tl.where(condition, -c + d * e, c)
    
    # Always: a[i] = b[i] + c[i] * d[i]
    a_new = b_new + c_new * d
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(c_ptr + offsets, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c