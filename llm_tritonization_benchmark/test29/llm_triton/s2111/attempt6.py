import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for diag in range(2, 2 * N):
        j_vals = diag - i_offsets
        
        valid_mask = (j_vals >= 1) & (j_vals < N) & (i_offsets >= 1) & (i_offsets < N)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            current_indices = j_vals * N + i_offsets
            left_indices = j_vals * N + (i_offsets - 1)  
            top_indices = (j_vals - 1) * N + i_offsets
            
            current_vals = tl.load(aa_ptr + current_indices, mask=valid_mask, other=0.0)
            left_vals = tl.load(aa_ptr + left_indices, mask=valid_mask, other=0.0)
            top_vals = tl.load(aa_ptr + top_indices, mask=valid_mask, other=0.0)
            
            new_vals = (left_vals + top_vals) / 1.9
            
            tl.store(aa_ptr + current_indices, new_vals, mask=valid_mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 512
    
    grid = (1,)
    s2111_kernel[grid](aa, N, BLOCK_SIZE)
    
    return aa