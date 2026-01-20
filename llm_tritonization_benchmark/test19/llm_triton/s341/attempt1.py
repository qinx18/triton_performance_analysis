import triton
import triton.language as tl
import torch

@triton.jit
def s341_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one block at a time sequentially
    # to maintain the stream compaction property
    block_id = tl.program_id(0)
    
    if block_id > 0:
        return
    
    # Initialize counter
    j = 0
    
    # Process all elements sequentially in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
                
            element_mask = (offsets == i)
            b_val = tl.sum(tl.where(element_mask, b_vals, 0.0))
            
            # Check condition and pack if positive
            if b_val > 0.0:
                tl.store(a_ptr + j, b_val)
                j += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (1,)  # Single thread block to maintain sequential processing
    
    s341_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )