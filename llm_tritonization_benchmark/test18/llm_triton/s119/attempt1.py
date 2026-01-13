import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Sequential processing due to wavefront dependencies
    # Each element depends on aa[i-1][j-1], creating diagonal dependencies
    
    # Process elements sequentially to maintain dependencies
    for i in range(1, N):
        for j in range(1, N):
            # Calculate linear indices
            current_idx = i * N + j
            prev_idx = (i - 1) * N + (j - 1)
            
            # Load values
            current_bb = tl.load(bb_ptr + current_idx)
            prev_aa = tl.load(aa_ptr + prev_idx)
            
            # Compute and store
            result = prev_aa + current_bb
            tl.store(aa_ptr + current_idx, result)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Launch kernel with single thread since we need sequential processing
    # due to wavefront dependencies
    BLOCK_SIZE = 1
    grid = (1,)
    
    s119_kernel[grid](
        aa, bb, N, BLOCK_SIZE
    )
    
    return aa