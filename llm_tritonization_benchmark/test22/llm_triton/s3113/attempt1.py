import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element's absolute value
    first_val = tl.load(a_ptr)
    max_val = tl.abs(first_val)
    
    # Process array in blocks
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Update max across the block
        block_max = tl.max(abs_vals)
        max_val = tl.maximum(max_val, block_max)
    
    return max_val

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread to handle the reduction
    grid = (1,)
    result = s3113_kernel[grid](a, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result