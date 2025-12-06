import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_read_offsets = (j - 1) * LEN_2D + i_idx
        aa_vals = tl.load(aa_ptr + aa_read_offsets, mask=i_mask, other=0.0)
        
        cc_read_offsets = j * LEN_2D + i_idx
        cc_vals = tl.load(cc_ptr + cc_read_offsets, mask=i_mask, other=0.0)
        
        result_vals = aa_vals + cc_vals
        
        aa_write_offsets = j * LEN_2D + i_idx
        tl.store(aa_ptr + aa_write_offsets, result_vals, mask=i_mask)
        
        # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        bb_read_offsets = (i_idx - 1) * LEN_2D + j
        bb_vals = tl.load(bb_ptr + bb_read_offsets, mask=i_mask, other=0.0)
        
        cc_read_offsets2 = i_idx * LEN_2D + j
        cc_vals2 = tl.load(cc_ptr + cc_read_offsets2, mask=i_mask, other=0.0)
        
        result_vals2 = bb_vals + cc_vals2
        
        bb_write_offsets = i_idx * LEN_2D + j
        tl.store(bb_ptr + bb_write_offsets, result_vals2, mask=i_mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    i_size = LEN_2D - 1
    grid = (triton.cdiv(i_size, BLOCK_SIZE),)
    
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )