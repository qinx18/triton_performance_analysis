import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    running_sum = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block of a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within the block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                # Add to running sum
                running_sum += tl.load(a_ptr + block_start + i)
                # Store running sum to b
                tl.store(b_ptr + block_start + i, running_sum)

def s3112_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    grid = (1,)  # Single thread program due to sequential dependency
    
    s3112_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return b[-1]  # Return final sum