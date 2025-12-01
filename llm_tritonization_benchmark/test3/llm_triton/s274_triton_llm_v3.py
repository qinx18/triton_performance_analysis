import torch
import triton
import triton.language as tl

@triton.jit
def s274_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    c_val = tl.load(c_ptr + offsets, mask=mask)
    d_val = tl.load(d_ptr + offsets, mask=mask)
    e_val = tl.load(e_ptr + offsets, mask=mask)
    b_val = tl.load(b_ptr + offsets, mask=mask)
    
    # First computation: a[i] = c[i] + e[i] * d[i]
    a_val = c_val + e_val * d_val
    
    # Conditional logic
    condition = a_val > 0.0
    
    # If a[i] > 0: b[i] = a[i] + b[i], a[i] stays the same
    # Else: a[i] = d[i] * e[i], b[i] stays the same
    new_b = tl.where(condition, a_val + b_val, b_val)
    new_a = tl.where(condition, a_val, d_val * e_val)
    
    # Store results
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(b_ptr + offsets, new_b, mask=mask)

def s274_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s274_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )