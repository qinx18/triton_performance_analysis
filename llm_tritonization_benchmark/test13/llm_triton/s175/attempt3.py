import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        # Calculate actual indices for this block
        indices = block_start + offsets
        
        # Filter indices to match the stride pattern: i % inc == 0
        valid_stride = (indices % inc) == 0
        
        # Check bounds: i < n - 1
        valid_bounds = indices < (n - 1)
        
        # Combine conditions
        mask = valid_stride & valid_bounds
        
        # Load b[i] values
        b_vals = tl.load(b_ptr + indices, mask=mask)
        
        # Load a[i + inc] values
        a_read_indices = indices + inc
        a_read_mask = mask & (a_read_indices < n)
        a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask)
        
        # Compute a[i] = a[i + inc] + b[i]
        result = a_vals + b_vals
        
        # Store result to a[i]
        tl.store(a_ptr + indices, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size to cover all possible indices
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s175_kernel[grid](a, b, inc, n, BLOCK_SIZE)