import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes sequentially since j depends on previous iterations
    # We'll use a single thread block to maintain the sequential nature
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j = -1
    
    # Process in blocks but maintain sequential order
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block of a
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                element_offset = block_start + i
                a_val = tl.load(a_ptr + element_offset)
                
                if a_val > 0.0:
                    j = j + 1
                    b_val = tl.load(b_ptr + j)
                    tl.store(a_ptr + element_offset, b_val)

def s342_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Launch with single thread block to maintain sequential processing
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a