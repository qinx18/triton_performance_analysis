import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for diag in range(2, 2 * N):
        j_vals = diag - i_offsets
        i_vals = i_offsets
        
        valid_mask = (j_vals >= 1) & (j_vals < N) & (i_vals >= 1) & (i_vals < N)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            left_offsets = j_vals * N + (i_vals - 1)
            top_offsets = (j_vals - 1) * N + i_vals
            
            left_vals = tl.load(aa_ptr + left_offsets, mask=valid_mask, other=0.0)
            top_vals = tl.load(aa_ptr + top_offsets, mask=valid_mask, other=0.0)
            
            result = (left_vals + top_vals) / 1.9
            
            result_offsets = j_vals * N + i_vals
            tl.store(aa_ptr + result_offsets, result, mask=valid_mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 256
    
    grid = (1,)
    s2111_kernel[grid](aa, N, BLOCK_SIZE)
    
    return aa