import torch
import triton
import triton.language as tl

@triton.jit
def s311_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    sum_val = 0.0
    
    # Process all elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from array a
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Sum the values in this block
        sum_val += tl.sum(vals, axis=0)
    
    # Store the final sum
    tl.store(sum_ptr, sum_val)

def s311_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor for sum
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program
    grid = (1,)
    s311_kernel[grid](
        a, sum_result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result