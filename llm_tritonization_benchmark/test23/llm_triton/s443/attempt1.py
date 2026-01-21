import triton
import triton.language as tl
import torch

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Condition: d[i] <= 0
    condition = d_vals <= 0.0
    
    # Compute both branches
    branch1 = b_vals * c_vals  # d[i] <= 0: multiply b and c
    branch2 = b_vals * b_vals  # d[i] > 0: multiply b and b
    
    # Select based on condition
    result = tl.where(condition, branch1, branch2)
    
    # Add to a[i]
    a_vals = a_vals + result
    
    # Store result
    tl.store(a_ptr + idx, a_vals, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)