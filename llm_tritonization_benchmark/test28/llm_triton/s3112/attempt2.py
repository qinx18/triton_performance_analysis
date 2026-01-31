import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n):
    BLOCK_SIZE = 128
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # This is a prefix sum operation - process sequentially
    sum_val = 0.0
    
    # Process array in blocks
    for block_start in tl.static_range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process elements sequentially within block
        for i in tl.static_range(BLOCK_SIZE):
            if block_start + i < n:
                # Get the value for this position
                val = tl.load(a_ptr + (block_start + i))
                sum_val += val
                # Store the running sum
                tl.store(b_ptr + (block_start + i), sum_val)

def s3112_triton(a, b):
    n = a.shape[0]
    
    # Launch kernel with single program to maintain sequential order
    s3112_kernel[(1,)](a, b, n)
    
    # Return final sum (last element of b)
    return b[-1].item()