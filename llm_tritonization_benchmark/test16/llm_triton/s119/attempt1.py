import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Sequential wavefront computation - no parallelization due to dependencies
    for i in range(1, N):
        for j in range(1, N):
            # Load aa[i-1, j-1]
            aa_prev_idx = (i - 1) * N + (j - 1)
            aa_prev_val = tl.load(aa_ptr + aa_prev_idx)
            
            # Load bb[i, j]
            bb_idx = i * N + j
            bb_val = tl.load(bb_ptr + bb_idx)
            
            # Compute and store aa[i, j]
            result = aa_prev_val + bb_val
            aa_idx = i * N + j
            tl.store(aa_ptr + aa_idx, result)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Ensure contiguous tensors
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    # Launch kernel with single thread since computation is sequential
    grid = (1,)
    s119_kernel[grid](
        aa, bb, N, BLOCK_SIZE=1
    )
    
    return aa