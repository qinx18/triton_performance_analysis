import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential loop processing blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Process each element in the block sequentially
        for local_i in range(BLOCK_SIZE):
            global_i = block_start + local_i
            if global_i >= n_elements:
                break
                
            # Calculate im1 and im2 based on current position
            if global_i == 0:
                im1 = n_elements - 1
                im2 = n_elements - 2
            elif global_i == 1:
                im1 = 0
                im2 = n_elements - 1
            else:
                im1 = global_i - 1
                im2 = global_i - 2
            
            # Load b[i], b[im1], b[im2]
            b_i = tl.load(b_ptr + global_i)
            b_im1 = tl.load(b_ptr + im1)
            b_im2 = tl.load(b_ptr + im2)
            
            # Compute and store result
            result = (b_i + b_im1 + b_im2) * 0.333
            tl.store(a_ptr + global_i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a