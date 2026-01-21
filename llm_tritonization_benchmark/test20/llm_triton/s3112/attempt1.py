import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize running sum
    running_sum = 0.0
    
    # Process in blocks sequentially to maintain prefix sum
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load current block
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute prefix sum within block
        prefix_sums = vals
        for stride in range(1, BLOCK_SIZE):
            shifted = tl.zeros_like(prefix_sums)
            # Manually shift elements
            for i in range(BLOCK_SIZE):
                if i >= stride:
                    shifted = tl.where(offsets == i, 
                                     tl.sum(tl.where(offsets == i - stride, prefix_sums, 0.0)), 
                                     shifted)
            prefix_sums = prefix_sums + shifted
        
        # Add running sum to prefix sums
        final_sums = prefix_sums + running_sum
        
        # Store results
        tl.store(b_ptr + current_offsets, final_sums, mask=mask)
        
        # Update running sum for next block
        if block_start + BLOCK_SIZE <= n:
            running_sum = running_sum + tl.sum(vals)
        else:
            # Handle partial last block
            valid_vals = tl.where(mask, vals, 0.0)
            running_sum = running_sum + tl.sum(valid_vals)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread block to maintain sequential dependency
    grid = (1,)
    s3112_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return the final sum (last element of b)
    return b[-1].item()