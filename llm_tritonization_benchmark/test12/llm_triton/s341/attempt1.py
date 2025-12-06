import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Stream compaction - pack positive values from b into a
    # This is inherently sequential, so we use a single thread block
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    j = 0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            
            element_mask = tl.arange(0, BLOCK_SIZE) == i
            b_val = tl.sum(tl.where(element_mask, b_vals, 0.0))
            
            # Check if this element is positive
            if b_val > 0.0:
                # Store at position j and increment j
                tl.store(a_ptr + j, b_val)
                j += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch single thread block since this is inherently sequential
    grid = (1,)
    
    s341_kernel[grid](
        b, a, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )