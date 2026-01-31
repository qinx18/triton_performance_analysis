import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Find early exit condition on CPU
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        # Find first index where exit condition is true
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before the exit point
        if exit_idx > 0:
            BLOCK_SIZE = 256
            grid = (triton.cdiv(exit_idx, BLOCK_SIZE),)
            
            s481_kernel[grid](
                a[:exit_idx], b[:exit_idx], c[:exit_idx],
                exit_idx, BLOCK_SIZE
            )
        # Exit early - don't process remaining elements
        return
    
    # No exit condition found, process all elements
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c,
        n_elements, BLOCK_SIZE
    )