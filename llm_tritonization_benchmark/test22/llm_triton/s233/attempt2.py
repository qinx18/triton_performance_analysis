import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        # First computation: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_prev_ptrs = aa_ptr + (j - 1) * LEN_2D + i_idx
        aa_curr_ptrs = aa_ptr + j * LEN_2D + i_idx
        cc_ptrs = cc_ptr + j * LEN_2D + i_idx
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        aa_new_vals = aa_prev_vals + cc_vals
        
        tl.store(aa_curr_ptrs, aa_new_vals, mask=i_mask)
    
    for j in range(1, LEN_2D):
        # Second computation: bb[j][i] = bb[j][i-1] + cc[j][i]
        bb_prev_idx = j * LEN_2D + (i_idx - 1)
        bb_curr_idx = j * LEN_2D + i_idx
        cc_idx = j * LEN_2D + i_idx
        
        bb_prev_ptrs = bb_ptr + bb_prev_idx
        bb_curr_ptrs = bb_ptr + bb_curr_idx
        cc_ptrs = cc_ptr + cc_idx
        
        prev_mask = (i_idx - 1) >= 0
        combined_mask = i_mask & prev_mask
        
        bb_prev_vals = tl.load(bb_prev_ptrs, mask=combined_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        bb_new_vals = bb_prev_vals + cc_vals
        
        tl.store(bb_curr_ptrs, bb_new_vals, mask=i_mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )