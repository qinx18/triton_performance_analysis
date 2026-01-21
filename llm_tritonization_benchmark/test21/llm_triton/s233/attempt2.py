import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, 
                LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1  # Start from i=1
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):  # Sequential j loop
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_read_offsets = (j - 1) * LEN_2D + i_idx
        cc_read_offsets = j * LEN_2D + i_idx
        aa_write_offsets = j * LEN_2D + i_idx
        
        aa_prev = tl.load(aa_ptr + aa_read_offsets, mask=i_mask, other=0.0)
        cc_val = tl.load(cc_ptr + cc_read_offsets, mask=i_mask, other=0.0)
        aa_new = aa_prev + cc_val
        
        tl.store(aa_ptr + aa_write_offsets, aa_new, mask=i_mask)
        
        # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
        i_idx_minus_1 = i_idx - 1
        bb_read_mask = i_mask & (i_idx_minus_1 >= 0)
        bb_read_offsets = j * LEN_2D + i_idx_minus_1
        bb_write_offsets = j * LEN_2D + i_idx
        
        bb_prev = tl.load(bb_ptr + bb_read_offsets, mask=bb_read_mask, other=0.0)
        bb_new = bb_prev + cc_val
        
        tl.store(bb_ptr + bb_write_offsets, bb_new, mask=i_mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Parallelize over i dimension (excluding i=0)
    i_size = LEN_2D - 1
    grid = (triton.cdiv(i_size, BLOCK_SIZE),)
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )