import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n):
    BLOCK_SIZE = 1024
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize running sum
    running_sum = tl.zeros((1,), dtype=tl.float32)
    
    # Process array in blocks sequentially
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load block of data
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element sequentially within block
        for i in range(BLOCK_SIZE):
            elem_idx = block_start + i
            if elem_idx < n:
                # Add current element to running sum
                current_val = vals[i]
                running_sum = running_sum + current_val
                
                # Store running sum to output
                tl.store(b_ptr + elem_idx, running_sum)

def s3112_triton(a, b):
    n = a.shape[0]
    
    # Launch kernel with single program to maintain sequential order
    s3112_kernel[(1,)](a, b, n)
    
    # Return final sum (last element of b)
    return b[-1].item()