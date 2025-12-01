import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Pack positive values - sequential algorithm
    # This cannot be effectively parallelized due to data dependencies
    
    # Process sequentially in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    j = 0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i >= n:
                break
            
            b_val = tl.load(b_ptr + block_start + i)
            if b_val > 0.0:
                tl.store(a_ptr + j, b_val)
                j += 1

def s341_triton(a, b):
    n = b.shape[0]
    BLOCK_SIZE = 256
    
    # Reset output array
    a.zero_()
    
    # Launch single thread since this is inherently sequential
    grid = (1,)
    s341_kernel[grid](b, a, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a