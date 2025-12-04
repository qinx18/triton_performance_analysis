import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    valid_i = (i_idx >= 1) & (i_idx < LEN_2D)
    
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_read_offsets = (j - 1) * LEN_2D + i_idx
        aa_write_offsets = j * LEN_2D + i_idx
        cc_offsets = j * LEN_2D + i_idx
        
        aa_vals = tl.load(aa_ptr + aa_read_offsets, mask=valid_i, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=valid_i, other=0.0)
        result_aa = aa_vals + cc_vals
        tl.store(aa_ptr + aa_write_offsets, result_aa, mask=valid_i)
        
        # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        bb_read_offsets = (i_idx - 1) * LEN_2D + j
        bb_write_offsets = i_idx * LEN_2D + j
        cc_offsets_bb = i_idx * LEN_2D + j
        
        bb_vals = tl.load(bb_ptr + bb_read_offsets, mask=valid_i, other=0.0)
        cc_vals_bb = tl.load(cc_ptr + cc_offsets_bb, mask=valid_i, other=0.0)
        result_bb = bb_vals + cc_vals_bb
        tl.store(bb_ptr + bb_write_offsets, result_bb, mask=valid_i)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )