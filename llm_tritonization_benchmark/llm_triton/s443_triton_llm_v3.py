import torch
import triton
import triton.language as tl

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load data
    d_vals = tl.load(d_ptr + indices, mask=mask, other=1.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Compute condition
    condition = d_vals <= 0.0
    
    # Compute both branches
    branch1 = b_vals * c_vals  # L20: b[i] * c[i]
    branch2 = b_vals * b_vals  # L30: b[i] * b[i]
    
    # Select based on condition
    result = tl.where(condition, branch1, branch2)
    
    # Update a[i] += result
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )