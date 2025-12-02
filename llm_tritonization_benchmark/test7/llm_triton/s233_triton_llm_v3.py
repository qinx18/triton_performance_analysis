import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get column index
    col_idx = tl.program_id(0)
    
    if col_idx >= LEN_2D - 1:
        return
    
    # Actual column (offset by 1)
    i = col_idx + 1
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, LEN_2D):
        aa_curr = tl.load(aa_ptr + j * LEN_2D + i)
        aa_prev = tl.load(aa_ptr + (j-1) * LEN_2D + i)
        cc_val = tl.load(cc_ptr + j * LEN_2D + i)
        result = aa_prev + cc_val
        tl.store(aa_ptr + j * LEN_2D + i, result)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j in range(1, LEN_2D):
        bb_prev = tl.load(bb_ptr + j * LEN_2D + (i-1))
        cc_val = tl.load(cc_ptr + j * LEN_2D + i)
        result = bb_prev + cc_val
        tl.store(bb_ptr + j * LEN_2D + i, result)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Launch kernel for columns 1 to LEN_2D-1
    grid = (LEN_2D - 1,)
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )