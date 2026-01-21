import triton
import triton.language as tl
import torch

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Condition: c[i] < 0
    condition = c_vals < 0.0
    
    # if c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    a_result = tl.where(condition, a_vals, c_vals + d_vals * e_vals)
    
    # if c[i] < 0: b[i] = a[i] + d[i] * d[i]
    b_result = a_vals + d_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_result, mask=mask)
    tl.store(b_ptr + offsets, b_result, mask=(mask & condition))

def s1161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1161_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)