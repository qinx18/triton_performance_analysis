import triton
import triton.language as tl
import torch

@triton.jit
def s443_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Conditional computation: if d[i] <= 0, use b[i] * c[i], else use b[i] * b[i]
    condition = d_vals <= 0.0
    result = tl.where(condition, b_vals * c_vals, b_vals * b_vals)
    
    # Update a[i]
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )