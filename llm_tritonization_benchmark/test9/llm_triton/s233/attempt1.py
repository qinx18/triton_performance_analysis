import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, LEN_2D):
        aa_val = tl.load(aa_ptr + (j-1) * LEN_2D + i)
        cc_val = tl.load(cc_ptr + j * LEN_2D + i)
        result = aa_val + cc_val
        tl.store(aa_ptr + j * LEN_2D + i, result)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j in range(1, LEN_2D):
        bb_val = tl.load(bb_ptr + j * LEN_2D + (i-1))
        cc_val = tl.load(cc_ptr + j * LEN_2D + i)
        result = bb_val + cc_val
        tl.store(bb_ptr + j * LEN_2D + i, result)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Launch kernel for each i from 1 to LEN_2D-1
    grid = (LEN_2D - 1,)
    BLOCK_SIZE = 256
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )