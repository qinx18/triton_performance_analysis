import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential computation for each block
    for block_idx in range(tl.cdiv(n_elements, BLOCK_SIZE)):
        start_idx = block_idx * BLOCK_SIZE
        
        for local_i in range(BLOCK_SIZE):
            i = start_idx + local_i
            if i >= n_elements:
                break
                
            # Calculate im1 and im2 based on current i
            if i == 0:
                im1 = n_elements - 1
                im2 = n_elements - 2
            elif i == 1:
                im1 = 0
                im2 = n_elements - 1
            else:
                im1 = i - 1
                im2 = i - 2
            
            # Load values
            b_i = tl.load(b_ptr + i)
            b_im1 = tl.load(b_ptr + im1)
            b_im2 = tl.load(b_ptr + im2)
            
            # Compute result
            result = (b_i + b_im1 + b_im2) * 0.333
            
            # Store result
            tl.store(a_ptr + i, result)

def s292_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single block to maintain sequential dependency
    grid = (1,)
    s292_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return a