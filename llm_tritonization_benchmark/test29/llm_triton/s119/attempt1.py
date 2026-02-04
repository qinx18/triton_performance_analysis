import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for diag in range(2, 2 * N):
        i = diag - j_offsets
        mask = (i >= 1) & (i < N) & (j_offsets >= 1) & (j_offsets < N)
        
        aa_idx = i * N + j_offsets
        aa_prev_idx = (i - 1) * N + (j_offsets - 1)
        bb_idx = i * N + j_offsets
        
        aa_prev_val = tl.load(aa_ptr + aa_prev_idx, mask=mask, other=0.0)
        bb_val = tl.load(bb_ptr + bb_idx, mask=mask, other=0.0)
        
        result = aa_prev_val + bb_val
        
        tl.store(aa_ptr + aa_idx, result, mask=mask)
        tl.debug_barrier()

def s119_triton(aa, bb, len_2d):
    N = len_2d
    grid = (1,)
    BLOCK_SIZE = 256
    
    s119_kernel[grid](aa, bb, N, BLOCK_SIZE)