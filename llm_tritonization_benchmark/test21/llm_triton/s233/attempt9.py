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
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_prev_offsets = (j - 1) * LEN_2D + i_idx
        cc_offsets = j * LEN_2D + i_idx
        aa_write_offsets = j * LEN_2D + i_idx
        
        aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=i_mask, other=0.0)
        cc_val = tl.load(cc_ptr + cc_offsets, mask=i_mask, other=0.0)
        aa_new = aa_prev + cc_val
        tl.store(aa_ptr + aa_write_offsets, aa_new, mask=i_mask)
        
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        bb_prev_mask = i_mask & (i_idx > 1)
        bb_prev_offsets = j * LEN_2D + (i_idx - 1)
        bb_write_offsets = j * LEN_2D + i_idx
        
        bb_prev = tl.load(bb_ptr + bb_prev_offsets, mask=bb_prev_mask, other=0.0)
        bb_new = bb_prev + cc_val
        
        # For i=1, we need special handling
        bb_first_mask = i_mask & (i_idx == 1)
        bb_first_offsets = j * LEN_2D + 0
        bb_first_val = tl.load(bb_ptr + bb_first_offsets, mask=bb_first_mask, other=0.0)
        bb_new = tl.where(bb_first_mask, bb_first_val + cc_val, bb_new)
        
        tl.store(bb_ptr + bb_write_offsets, bb_new, mask=i_mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    i_size = LEN_2D - 1
    grid = (triton.cdiv(i_size, BLOCK_SIZE),)
    
    s233_kernel[grid](aa, bb, cc, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)