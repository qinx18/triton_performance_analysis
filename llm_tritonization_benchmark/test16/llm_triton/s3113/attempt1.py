import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max with first element
    first_val = tl.load(a_ptr)
    block_max = tl.abs(first_val)
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute absolute values and find max in this block
        abs_vals = tl.abs(vals)
        block_local_max = tl.max(abs_vals)
        
        # Update global max
        block_max = tl.maximum(block_max, block_local_max)
    
    # Store result at the first position
    tl.store(a_ptr, block_max)

def s3113_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Create a copy to avoid modifying input
    a_copy = a.clone()
    
    # Launch kernel with single program
    s3113_kernel[(1,)](
        a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the result stored at first position
    return a_copy[0].item()