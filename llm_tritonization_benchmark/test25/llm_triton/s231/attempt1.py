import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    for j in range(1, len_2d):
        # Load aa[j-1][i] and bb[j][i]
        aa_prev_ptrs = aa_ptr + (j - 1) * len_2d + i_idx
        bb_ptrs = bb_ptr + j * len_2d + i_idx
        aa_curr_ptrs = aa_ptr + j * len_2d + i_idx
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask)
        bb_vals = tl.load(bb_ptrs, mask=i_mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = aa_prev_vals + bb_vals
        
        # Store result
        tl.store(aa_curr_ptrs, result, mask=i_mask)

def s231_triton(aa, bb, len_2d):
    BLOCK_SIZE = 32
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )