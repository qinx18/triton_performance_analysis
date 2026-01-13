import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load data
    d_vals = tl.load(d_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute conditions
    cond1 = d_vals < 0.0
    cond2 = d_vals == 0.0
    
    # Compute results for each condition
    result1 = b_vals * c_vals
    result2 = b_vals * b_vals
    result3 = c_vals * c_vals
    
    # Select based on conditions
    temp = tl.where(cond2, result2, result3)
    final_result = tl.where(cond1, result1, temp)
    
    # Update a
    new_a = a_vals + final_result
    tl.store(a_ptr + idx, new_a, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)