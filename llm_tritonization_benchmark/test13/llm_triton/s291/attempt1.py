import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    # Handle the case where we need multiple blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        if block_start > block_id * BLOCK_SIZE:
            break
        if block_start < block_id * BLOCK_SIZE:
            continue
            
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Sequential implementation to preserve dependencies
        for i in range(BLOCK_SIZE):
            idx = block_start + i
            if idx >= n_elements:
                break
                
            # Get im1 value (previous index, wrapping around)
            im1_idx = n_elements - 1 if idx == 0 else idx - 1
            
            # Load values
            b_i = tl.load(b_ptr + idx)
            b_im1 = tl.load(b_ptr + im1_idx)
            
            # Compute and store
            result = (b_i + b_im1) * 0.5
            tl.store(a_ptr + idx, result)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block to handle sequential dependencies
    grid = (1,)
    
    s291_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a