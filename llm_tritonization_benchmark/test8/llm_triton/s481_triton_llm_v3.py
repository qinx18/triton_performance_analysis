import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    
    # Check if any element is negative (exit condition)
    negative_mask = d_vals < 0.0
    has_negative = tl.sum(negative_mask.to(tl.int32)) > 0
    
    # If no negative values, proceed with computation
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Check for negative values first (exit condition)
    if torch.any(d < 0.0):
        return  # Exit early if any negative values found
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )