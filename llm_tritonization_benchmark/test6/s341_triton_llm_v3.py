import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Pack positive values from b into a (sequential packing)
    # This is inherently sequential, so we process one block at a time
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Sequential processing - cannot be parallelized due to unknown j values
    offsets = tl.arange(0, BLOCK_SIZE)
    j = 0  # Start from 0 (equivalent to j = -1; j++)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            
            # Load single element
            b_val = tl.load(b_ptr + block_start + i)
            
            # Check if positive and pack
            if b_val > 0.0:
                tl.store(a_ptr + j, b_val)
                j += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Zero out the output array
    a.zero_()
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block processing due to sequential nature
    
    s341_kernel[grid](
        b, a, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a