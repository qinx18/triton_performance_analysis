import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < LEN_2D
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, LEN_2D):
        aa_prev_ptrs = aa_ptr + (j - 1) * LEN_2D + i_idx
        aa_curr_ptrs = aa_ptr + j * LEN_2D + i_idx
        cc_ptrs = cc_ptr + j * LEN_2D + i_idx
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        aa_new_vals = aa_prev_vals + cc_vals
        tl.store(aa_curr_ptrs, aa_new_vals, mask=i_mask)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j in range(1, LEN_2D):
        cc_ptrs = cc_ptr + j * LEN_2D + i_idx
        cc_vals = tl.load(cc_ptrs, mask=i_mask, other=0.0)
        
        # For i=1, we need bb[j][0]
        bb_prev_idx = i_idx - 1
        bb_prev_mask = (i_idx > 0) & (i_idx < LEN_2D)
        bb_prev_ptrs = bb_ptr + j * LEN_2D + bb_prev_idx
        bb_curr_ptrs = bb_ptr + j * LEN_2D + i_idx
        
        bb_prev_vals = tl.load(bb_prev_ptrs, mask=bb_prev_mask, other=0.0)
        bb_new_vals = bb_prev_vals + cc_vals
        tl.store(bb_curr_ptrs, bb_new_vals, mask=i_mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    i_size = LEN_2D - 1
    grid = (triton.cdiv(i_size, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )