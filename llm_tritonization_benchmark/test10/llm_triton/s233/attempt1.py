import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_read_offsets = (j - 1) * LEN_2D + i_idx
        cc_read_offsets = j * LEN_2D + i_idx
        aa_write_offsets = j * LEN_2D + i_idx
        
        aa_vals = tl.load(aa_ptr + aa_read_offsets, mask=i_mask)
        cc_vals = tl.load(cc_ptr + cc_read_offsets, mask=i_mask)
        result_aa = aa_vals + cc_vals
        tl.store(aa_ptr + aa_write_offsets, result_aa, mask=i_mask)
        
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        bb_read_offsets = j * LEN_2D + (i_idx - 1)
        bb_write_offsets = j * LEN_2D + i_idx
        
        bb_vals = tl.load(bb_ptr + bb_read_offsets, mask=i_mask)
        result_bb = bb_vals + cc_vals
        tl.store(bb_ptr + bb_write_offsets, result_bb, mask=i_mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )