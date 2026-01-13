import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process each block sequentially due to dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # For each element in the block, compute with proper wrap-around
        for local_i in range(BLOCK_SIZE):
            global_i = block_start + local_i
            if global_i >= n_elements:
                break
                
            # Compute im1 and im2 indices based on current position
            if global_i == 0:
                im1_idx = n_elements - 1
                im2_idx = n_elements - 2
            elif global_i == 1:
                im1_idx = 0
                im2_idx = n_elements - 1
            else:
                im1_idx = global_i - 1
                im2_idx = global_i - 2
            
            # Load required values
            b_i = tl.load(b_ptr + global_i)
            b_im1 = tl.load(b_ptr + im1_idx)
            b_im2 = tl.load(b_ptr + im2_idx)
            
            # Compute result
            result = (b_i + b_im1 + b_im2) * 0.333
            
            # Store result
            tl.store(a_ptr + global_i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(n_elements))
    
    # Launch kernel with single program since we need sequential processing
    s292_kernel[(1,)](a, b, n_elements, BLOCK_SIZE)
    
    return a