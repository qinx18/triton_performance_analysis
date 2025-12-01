import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the column index (i dimension)
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, LEN_2D):
        aa_curr_offset = j * LEN_2D + i
        aa_prev_offset = (j - 1) * LEN_2D + i
        cc_offset = j * LEN_2D + i
        
        aa_prev_val = tl.load(aa_ptr + aa_prev_offset)
        cc_val = tl.load(cc_ptr + cc_offset)
        result = aa_prev_val + cc_val
        tl.store(aa_ptr + aa_curr_offset, result)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j in range(1, LEN_2D):
        bb_curr_offset = j * LEN_2D + i
        bb_prev_offset = j * LEN_2D + (i - 1)
        cc_offset = j * LEN_2D + i
        
        bb_prev_val = tl.load(bb_ptr + bb_prev_offset)
        cc_val = tl.load(cc_ptr + cc_offset)
        result = bb_prev_val + cc_val
        tl.store(bb_ptr + bb_curr_offset, result)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per column (excluding first column)
    grid = (LEN_2D - 1,)
    BLOCK_SIZE = 1
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )