import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is a cumulative sum (prefix sum) operation
    # It must be computed sequentially, so we use a single thread
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process all elements sequentially in blocks
    running_sum = 0.0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load the block
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n:
                # Extract single element from vector
                elem_mask = offsets == i
                elem = tl.sum(tl.where(elem_mask, vals, 0.0))
                running_sum += elem
                # Store the cumulative sum
                tl.store(b_ptr + block_start + i, running_sum)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single thread since this is inherently sequential
    grid = (1,)
    s3112_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return the final sum (last element of b)
    return b[-1].item()