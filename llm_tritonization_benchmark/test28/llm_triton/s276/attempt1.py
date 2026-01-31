import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, mid, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Condition: i+1 < mid (where i is the index)
    condition_mask = (indices + 1) < mid
    
    # Compute result based on condition
    result_c = a_vals + b_vals * c_vals
    result_d = a_vals + b_vals * d_vals
    
    # Select based on condition
    result = tl.where(condition_mask, result_c, result_d)
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        mid,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )