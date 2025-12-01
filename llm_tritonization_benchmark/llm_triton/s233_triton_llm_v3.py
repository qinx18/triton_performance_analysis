import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one column (i) at a time
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    # Must be sequential in j dimension due to dependency
    for j in range(1, LEN_2D):
        aa_curr_ptr = aa_ptr + j * LEN_2D + i
        aa_prev_ptr = aa_ptr + (j - 1) * LEN_2D + i
        cc_curr_ptr = cc_ptr + j * LEN_2D + i
        
        aa_prev_val = tl.load(aa_prev_ptr)
        cc_val = tl.load(cc_curr_ptr)
        result = aa_prev_val + cc_val
        tl.store(aa_curr_ptr, result)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    # Must be sequential in j dimension due to dependency on bb[j][i-1]
    for j in range(1, LEN_2D):
        bb_curr_ptr = bb_ptr + j * LEN_2D + i
        bb_prev_ptr = bb_ptr + j * LEN_2D + (i - 1)
        cc_curr_ptr = cc_ptr + j * LEN_2D + i
        
        bb_prev_val = tl.load(bb_prev_ptr)
        cc_val = tl.load(cc_curr_ptr)
        result = bb_prev_val + cc_val
        tl.store(bb_curr_ptr, result)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per column (excluding first column)
    grid = (LEN_2D - 1,)
    
    s233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=256
    )