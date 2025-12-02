import torch
import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Sequential loop over j dimension due to dependency
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] + bb[j][i]
        aa_prev = tl.load(aa_ptr + (j - 1) * LEN_2D + i)
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        result = aa_prev + bb_val
        
        # Store to aa[j][i]
        tl.store(aa_ptr + j * LEN_2D + i, result)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s231_kernel[grid](aa, bb, LEN_2D)
    
    return aa