import triton
import triton.language as tl
import torch

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Conditional computation: if d[i] <= 0, use b[i] * c[i], else use b[i] * b[i]
    condition = d_vals <= 0.0
    result = tl.where(condition, b_vals * c_vals, b_vals * b_vals)
    
    # Update a[i]
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + idx, a_vals, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )