import torch
import triton
import triton.language as tl

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute condition: i+1 < mid
    condition = (indices + 1) < mid
    
    # Conditional computation
    result_c = b_vals * c_vals
    result_d = b_vals * d_vals
    result = tl.where(condition, result_c, result_d)
    
    # Update a
    new_a = a_vals + result
    
    # Store result
    tl.store(a_ptr + indices, new_a, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](a, b, c, d, n_elements, mid, BLOCK_SIZE)