import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize running sum
    running_sum = 0.0
    
    # Process elements sequentially to maintain cumulative sum dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of elements from array a
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                running_sum += vals[i]
                # Store running sum to b at position i
                tl.store(b_ptr + block_start + i, running_sum)

def s3112_triton(a, b):
    n_elements = a.shape[0]
    
    # Use single thread for sequential dependency
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program instance
    s3112_kernel[(1,)](
        a, b, n_elements, BLOCK_SIZE
    )
    
    # Return the final sum (last element of b)
    return b[-1].item()