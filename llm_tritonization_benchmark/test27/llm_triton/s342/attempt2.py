import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Sequential processing - can't parallelize due to data dependency
    # j value depends on all previous iterations
    
    j = -1
    
    # Process elements sequentially in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load current block of a values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for i in range(BLOCK_SIZE):
            current_idx = block_start + i
            if current_idx < N:
                # Check if a[i] > 0
                a_val = tl.load(a_ptr + current_idx)
                if a_val > 0.0:
                    j = j + 1
                    b_val = tl.load(b_ptr + j)
                    tl.store(a_ptr + current_idx, b_val)

def s342_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single thread since this is inherently sequential
    grid = (1,)
    
    s342_kernel[grid](
        a, b, N, BLOCK_SIZE
    )
    
    return a