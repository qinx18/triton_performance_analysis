import torch
import triton
import triton.language as tl

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize running sum
    running_sum = 0.0
    
    # Process array in blocks
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values from array a
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially to maintain running sum
        for i in range(BLOCK_SIZE):
            if block_start + i < n:
                val = tl.load(a_ptr + block_start + i)
                running_sum += val
                tl.store(b_ptr + block_start + i, running_sum)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)  # Single thread block to maintain sequential dependency
    s3112_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return the final sum (last element of b)
    return b[-1].item()