import triton
import triton.language as tl
import torch

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get the starting position for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n
    
    # Load the constant value from the first element of the copy
    const_val = tl.load(a_copy_ptr)
    
    # Store the constant value to all positions in this block
    tl.store(a_ptr + offsets, const_val, mask=mask)

def s293_triton(a):
    n = a.shape[0]
    
    # Handle crossing threshold dependency
    threshold = 0
    
    # Save original value before it gets modified
    orig_a_at_threshold = a[threshold].clone()
    
    # Phase 1: i = 0 to threshold (uses original value)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(threshold + 1, BLOCK_SIZE),)
    s293_kernel[grid](a, orig_a_at_threshold, threshold + 1, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: i = threshold+1 to end (uses updated value)
    if threshold + 1 < n:
        remaining = n - (threshold + 1)
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        # Create a view for the remaining elements
        a_remaining = a[threshold + 1:]
        updated_a_at_threshold = a[threshold]
        s293_kernel[grid](a_remaining, updated_a_at_threshold, remaining, BLOCK_SIZE=BLOCK_SIZE)