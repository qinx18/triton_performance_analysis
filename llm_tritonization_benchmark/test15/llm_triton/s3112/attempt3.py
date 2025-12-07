import torch
import triton
import triton.language as tl

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Only use the first program to do all work sequentially
    if pid != 0:
        return
    
    # Initialize running sum as scalar
    running_sum = 0.0
    
    # Process elements sequentially
    for i in range(n_elements):
        # Load single element
        val = tl.load(a_ptr + i)
        running_sum = running_sum + val
        # Store running sum
        tl.store(b_ptr + i, running_sum)

def s3112_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single thread since this is inherently serial
    grid = (1,)
    
    s3112_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the final sum (last element of b)
    return b[-1].item()