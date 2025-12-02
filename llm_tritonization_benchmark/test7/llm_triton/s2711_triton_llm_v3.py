import torch
import triton
import triton.language as tl

@triton.jit
def s2711_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Conditional update: if b[i] != 0.0, then a[i] += b[i] * c[i]
    nonzero_mask = b_vals != 0.0
    update_vals = b_vals * c_vals
    result = tl.where(nonzero_mask, a_vals + update_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s2711_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2711_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)