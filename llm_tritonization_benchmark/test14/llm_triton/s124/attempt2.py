import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential processing due to dependency on induction variable j
    
    j = -1
    
    # Process all elements sequentially in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_block_size = min(BLOCK_SIZE, n_elements - block_start)
        
        # Load data for current block
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < current_block_size
        current_offsets = block_start + offsets
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each valid element
        for i in range(current_block_size):
            j += 1
            
            # Compute based on condition
            condition = b_vals[i] > 0.0
            if condition:
                result = b_vals[i] + d_vals[i] * e_vals[i]
            else:
                result = c_vals[i] + d_vals[i] * e_vals[i]
            
            # Store result at position j
            tl.store(a_ptr + j, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    
    # Use small block size for sequential processing
    BLOCK_SIZE = 32
    
    # Launch single thread to maintain sequential order
    grid = (1,)
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a