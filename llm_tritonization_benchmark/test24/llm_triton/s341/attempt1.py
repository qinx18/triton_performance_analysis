import triton
import triton.language as tl
import torch

@triton.jit
def s341_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the stream compaction sequentially
    # Each program handles a portion of the array
    pid = tl.program_id(axis=0)
    
    if pid > 0:
        return
    
    # Process sequentially to maintain correct packing order
    j = 0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            
            element_mask = i == offsets
            b_val = tl.sum(tl.where(element_mask, b_vals, 0.0))
            
            # Check condition and pack if true
            if b_val > 0.0:
                tl.store(a_ptr + j, b_val)
                j += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)  # Single program to maintain sequential order
    s341_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)