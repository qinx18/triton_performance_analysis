import torch
import triton
import triton.language as tl

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    d_vals = tl.load(d_ptr + idx, mask=mask, other=1.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    
    # Conditional computation: if d[i] <= 0.0 then a[i] += b[i] * c[i] else a[i] += b[i] * b[i]
    condition = d_vals <= 0.0
    result = tl.where(condition, a_vals + b_vals * c_vals, a_vals + b_vals * b_vals)
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )