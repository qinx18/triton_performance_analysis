import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially to maintain dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block of b
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Process each element in the block sequentially
        for local_i in range(BLOCK_SIZE):
            global_i = block_start + local_i
            mask_elem = global_i < n_elements
            
            tl.static_if(mask_elem):
                # Calculate im1 and im2 based on current position
                tl.static_if(global_i == 0):
                    im1 = n_elements - 1
                    im2 = n_elements - 2
                tl.static_elif(global_i == 1):
                    im1 = 0
                    im2 = n_elements - 1
                tl.static_else():
                    im1 = global_i - 1
                    im2 = global_i - 2
                
                # Load b values
                b_i = tl.load(b_ptr + global_i)
                b_im1 = tl.load(b_ptr + im1)
                b_im2 = tl.load(b_ptr + im2)
                
                # Compute result
                result = (b_i + b_im1 + b_im2) * 0.333
                
                # Store result
                tl.store(a_ptr + global_i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block to maintain sequential dependencies
    grid = (1,)
    
    s292_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )