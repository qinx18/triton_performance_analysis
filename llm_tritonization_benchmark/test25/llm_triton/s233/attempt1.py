import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < len_2d
    
    for j in range(1, len_2d):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_read_offsets = (j - 1) * len_2d + i_idx
        aa_write_offsets = j * len_2d + i_idx
        cc_offsets = j * len_2d + i_idx
        
        aa_prev = tl.load(aa_ptr + aa_read_offsets, mask=i_mask, other=0.0)
        cc_val = tl.load(cc_ptr + cc_offsets, mask=i_mask, other=0.0)
        aa_new = aa_prev + cc_val
        tl.store(aa_ptr + aa_write_offsets, aa_new, mask=i_mask)
        
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        bb_read_offsets = j * len_2d + (i_idx - 1)
        bb_write_offsets = j * len_2d + i_idx
        
        bb_prev = tl.load(bb_ptr + bb_read_offsets, mask=i_mask, other=0.0)
        bb_new = bb_prev + cc_val
        tl.store(bb_ptr + bb_write_offsets, bb_new, mask=i_mask)

def s233_triton(aa, bb, cc, len_2d):
    BLOCK_SIZE = 64
    i_size = len_2d - 1
    grid = (triton.cdiv(i_size, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )