import torch
import triton
import triton.language as tl

@triton.jit
def s2712_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Conditional update: if a[i] > b[i], then a[i] += b[i] * c[i]
    condition = a_vals > b_vals
    update_vals = b_vals * c_vals
    result = tl.where(condition, a_vals + update_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s2712_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2712_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)