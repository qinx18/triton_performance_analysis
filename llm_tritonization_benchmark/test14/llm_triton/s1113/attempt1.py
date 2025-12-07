import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(a_ptr, b_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Load a[threshold] value (broadcast to all elements)
        a_threshold_val = tl.load(a_ptr + threshold)
        
        # Compute result
        result = a_threshold_val + b_vals
        
        # Store to a
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s1113_triton(a, b):
    n = a.shape[0]
    threshold = n // 2
    
    # Create read-only copy before launching kernel
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1113_kernel[grid](
        a_copy,  # Read from copy
        b,
        n,
        threshold,
        BLOCK_SIZE
    )
    
    # Copy results back to original array
    a.copy_(a_copy)