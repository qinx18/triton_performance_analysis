import torch
import triton
import triton.language as tl

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Condition: d[i] <= 0
    condition = d_vals <= 0.0
    
    # Compute both branches
    branch1 = b_vals * c_vals  # d[i] <= 0: b[i] * c[i]
    branch2 = b_vals * b_vals  # d[i] > 0:  b[i] * b[i]
    
    # Select based on condition
    result = tl.where(condition, branch1, branch2)
    
    # Update a[i]
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)