import torch
import triton
import triton.language as tl

@triton.jit
def s2711_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    
    # Check if b[i] != 0.0
    non_zero_mask = b_vals != 0.0
    
    # Compute b[i] * c[i]
    product = b_vals * c_vals
    
    # Conditionally add to a[i] where b[i] != 0.0
    result = tl.where(non_zero_mask, a_vals + product, a_vals)
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s2711_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2711_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)