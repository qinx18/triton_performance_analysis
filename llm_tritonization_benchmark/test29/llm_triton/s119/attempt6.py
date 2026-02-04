import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for diag in range(2, 2 * N):
        i_vals = diag - j_offsets
        
        mask = (i_vals >= 1) & (i_vals < N) & (j_offsets >= 1) & (j_offsets < N)
        
        valid_i = tl.where(mask, i_vals, 0)
        valid_j = tl.where(mask, j_offsets, 0)
        
        read_i = tl.where(mask, i_vals - 1, 0)
        read_j = tl.where(mask, j_offsets - 1, 0)
        
        read_offsets = read_i * N + read_j
        write_offsets = valid_i * N + valid_j
        bb_offsets = valid_i * N + valid_j
        
        aa_vals = tl.load(aa_ptr + read_offsets, mask=mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
        
        result = aa_vals + bb_vals
        
        tl.store(aa_ptr + write_offsets, result, mask=mask)

def s119_triton(aa, bb, len_2d):
    N = len_2d
    BLOCK_SIZE = min(256, N)
    
    grid = (1,)
    s119_kernel[grid](aa, bb, N, BLOCK_SIZE)
    
    return aa