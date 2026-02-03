import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for diag in range(2, 2 * N):
        j_vals = diag - i_offsets
        
        mask = (j_vals >= 1) & (j_vals < N) & (i_offsets >= 1) & (i_offsets < N)
        
        if tl.sum(mask.to(tl.int32)) > 0:
            aa_idx = j_vals * N + i_offsets
            aa_left_idx = j_vals * N + (i_offsets - 1)
            aa_up_idx = (j_vals - 1) * N + i_offsets
            
            aa_left = tl.load(aa_ptr + aa_left_idx, mask=mask, other=0.0)
            aa_up = tl.load(aa_ptr + aa_up_idx, mask=mask, other=0.0)
            
            result = (aa_left + aa_up) / 1.9
            
            tl.store(aa_ptr + aa_idx, result, mask=mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 256
    
    grid = (1,)
    s2111_kernel[grid](aa, N, BLOCK_SIZE)