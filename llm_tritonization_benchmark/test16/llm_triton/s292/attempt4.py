import torch
import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values for current block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Process each element in the block
    for idx in range(BLOCK_SIZE):
        if block_start + idx < n_elements:
            i = block_start + idx
            
            # Calculate im1 and im2 based on the pattern from C code
            if i == 0:
                im1 = n_elements - 1
                im2 = n_elements - 2
            elif i == 1:
                im1 = 0
                im2 = n_elements - 1
            else:
                im1 = i - 1
                im2 = i - 2
            
            # Load required values
            b_i = tl.load(b_ptr + i)
            b_im1 = tl.load(b_ptr + im1)
            b_im2 = tl.load(b_ptr + im2)
            
            # Compute and store result
            result = (b_i + b_im1 + b_im2) * 0.333
            tl.store(a_ptr + i, result)

def s292_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_elements:
        BLOCK_SIZE *= 2
    
    # Use single thread block due to sequential dependencies
    grid = (1,)
    
    s292_kernel[grid](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a