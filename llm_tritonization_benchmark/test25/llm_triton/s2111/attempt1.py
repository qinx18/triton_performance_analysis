import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Sequential processing for wavefront pattern
    for j in range(1, N):
        for i in range(1, N):
            # Read dependencies: aa[j][i-1] and aa[j-1][i]
            left_val = tl.load(aa_ptr + j * N + (i - 1))
            top_val = tl.load(aa_ptr + (j - 1) * N + i)
            
            # Compute new value
            new_val = (left_val + top_val) / 1.9
            
            # Store result
            tl.store(aa_ptr + j * N + i, new_val)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 32
    
    # Use single thread for sequential computation
    grid = (1,)
    
    s2111_kernel[grid](
        aa,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa