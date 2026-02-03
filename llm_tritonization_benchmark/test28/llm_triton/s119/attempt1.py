import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for diag in range(2, 2 * N):
        i_vals = diag - j_offsets
        
        valid_mask = (i_vals >= 1) & (i_vals < N) & (j_offsets >= 1) & (j_offsets < N)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            aa_read_offsets = (i_vals - 1) * N + (j_offsets - 1)
            aa_write_offsets = i_vals * N + j_offsets
            bb_offsets = i_vals * N + j_offsets
            
            aa_prev = tl.load(aa_ptr + aa_read_offsets, mask=valid_mask, other=0.0)
            bb_vals = tl.load(bb_ptr + bb_offsets, mask=valid_mask, other=0.0)
            
            result = aa_prev + bb_vals
            
            tl.store(aa_ptr + aa_write_offsets, result, mask=valid_mask)

def s119_triton(aa, bb, len_2d):
    N = len_2d
    BLOCK_SIZE = 512
    
    grid = (1,)
    s119_kernel[grid](aa, bb, N, BLOCK_SIZE)
    
    return aa