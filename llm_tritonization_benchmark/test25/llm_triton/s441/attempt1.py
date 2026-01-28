import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Compute conditions
    cond1 = d_vals < 0.0
    cond2 = d_vals == 0.0
    
    # Compute results for each condition
    result1 = b_vals * c_vals
    result2 = b_vals * b_vals
    result3 = c_vals * c_vals
    
    # Select appropriate result based on conditions
    result = tl.where(cond1, result1, tl.where(cond2, result2, result3))
    
    # Update a
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + idx, a_vals, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)