import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Update a array
    a_vals = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s481_triton(a, b, c, d):
    n = a.shape[0]
    
    # Check for early exit condition
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            BLOCK_SIZE = 256
            grid = (triton.cdiv(exit_idx, BLOCK_SIZE),)
            s481_kernel[grid](a, b, c, d, exit_idx, BLOCK_SIZE)
        return
    
    # No exit condition, process all elements
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s481_kernel[grid](a, b, c, d, n, BLOCK_SIZE)