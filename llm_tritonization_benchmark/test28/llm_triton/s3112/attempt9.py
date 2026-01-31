import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n):
    BLOCK_SIZE = 1024
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize running sum
    running_sum = 0.0
    
    # Process array in blocks
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load block of a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in tl.static_range(BLOCK_SIZE):
            if block_start + i < n:
                # Add current element to running sum
                elem_val = tl.load(a_ptr + (block_start + i))
                running_sum += elem_val
                # Store running sum to b
                tl.store(b_ptr + (block_start + i), running_sum)

def s3112_triton(a, b):
    n = a.shape[0]
    
    # Launch kernel with single block since this is inherently sequential
    s3112_kernel[(1,)](a, b, n)
    
    # Return final sum (last element of b)
    return b[-1].item()