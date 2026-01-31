import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load d values to check exit condition
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=1.0)  # Use 1.0 to avoid false positive
    
    # Check if any element is negative (exit condition)
    exit_condition = d_vals < 0.0
    
    # If exit condition is met, we need to find the first occurrence
    # For now, we'll process all elements and let the wrapper handle early exit
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute a[i] += b[i] * c[i] only where exit condition is not met
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)
    
    # Return whether any exit condition was encountered
    # Note: This is a simplified approach - proper early exit requires host coordination

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # First, check for early exit condition on CPU
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        # Find first index where exit condition is true
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before the exit point
        if exit_idx > 0:
            a_slice = a[:exit_idx]
            b_slice = b[:exit_idx]
            c_slice = c[:exit_idx]
            d_slice = d[:exit_idx]
            
            BLOCK_SIZE = 256
            grid = (triton.cdiv(exit_idx, BLOCK_SIZE),)
            
            s481_kernel[grid](
                a_slice, b_slice, c_slice, d_slice,
                exit_idx, BLOCK_SIZE
            )
        # Exit early - don't process remaining elements
        return
    
    # No exit condition found, process all elements
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements, BLOCK_SIZE
    )