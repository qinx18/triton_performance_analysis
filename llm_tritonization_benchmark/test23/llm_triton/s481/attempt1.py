import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load d values to check exit condition
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Check if any d[i] < 0
    exit_condition = d_vals < 0.0
    if tl.any(exit_condition):
        # Find first index where exit condition is true
        exit_mask = tl.where(exit_condition, offsets, n_elements)
        min_exit_idx = tl.min(exit_mask)
        
        # Only process elements before the exit point
        valid_mask = mask & (offsets < min_exit_idx)
    else:
        valid_mask = mask
    
    # Load b and c values
    b_vals = tl.load(b_ptr + offsets, mask=valid_mask)
    c_vals = tl.load(c_ptr + offsets, mask=valid_mask)
    
    # Load current a values
    a_vals = tl.load(a_ptr + offsets, mask=valid_mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=valid_mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Check for early exit condition globally first
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            a[:exit_idx] += b[:exit_idx] * c[:exit_idx]
        # Exit here - no further processing
        return
    
    # No exit condition, process all elements with Triton kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )