import triton
import triton.language as tl
import torch

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Conditional computation: if d[i] <= 0, use b*c, else use b*b
    condition = d_vals <= 0.0
    term = tl.where(condition, b_vals * c_vals, b_vals * b_vals)
    
    # Update a[i] += term
    result = a_vals + term
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s443_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s443_kernel[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)