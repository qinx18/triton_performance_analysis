import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Process all elements sequentially since this is a wavefront pattern
    for i in range(1, N):
        for j in range(1, N):
            # Calculate linear indices
            current_idx = i * N + j
            prev_idx = (i - 1) * N + (j - 1)
            
            # Load values
            aa_prev = tl.load(aa_ptr + prev_idx)
            bb_val = tl.load(bb_ptr + current_idx)
            
            # Compute and store
            result = aa_prev + bb_val
            tl.store(aa_ptr + current_idx, result)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Launch single thread since we need sequential processing
    grid = (1,)
    BLOCK_SIZE = 256
    
    s119_kernel[grid](
        aa, bb, N, BLOCK_SIZE
    )
    
    return aa