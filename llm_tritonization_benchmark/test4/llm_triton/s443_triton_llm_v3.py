import torch
import triton
import triton.language as tl

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    a_ptrs = a_ptr + block_start + offsets
    b_ptrs = b_ptr + block_start + offsets
    c_ptrs = c_ptr + block_start + offsets
    d_ptrs = d_ptr + block_start + offsets
    
    a_vals = tl.load(a_ptrs, mask=mask)
    b_vals = tl.load(b_ptrs, mask=mask)
    c_vals = tl.load(c_ptrs, mask=mask)
    d_vals = tl.load(d_ptrs, mask=mask)
    
    # if (d[i] <= 0.0) then a[i] += b[i] * c[i] else a[i] += b[i] * b[i]
    condition = d_vals <= 0.0
    result = tl.where(condition, a_vals + b_vals * c_vals, a_vals + b_vals * b_vals)
    
    tl.store(a_ptrs, result, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)
    
    return a