import torch
import triton
import triton.language as tl

@triton.jit
def s2233_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, LEN_2D):
        # Load aa[j-1][i]
        aa_prev_offset = (j - 1) * LEN_2D + i
        aa_prev_val = tl.load(aa_ptr + aa_prev_offset)
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute and store aa[j][i]
        result = aa_prev_val + cc_val
        aa_offset = j * LEN_2D + i
        tl.store(aa_ptr + aa_offset, result)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    for j in range(1, LEN_2D):
        # Load bb[i-1][j]
        bb_prev_offset = (i - 1) * LEN_2D + j
        bb_prev_val = tl.load(bb_ptr + bb_prev_offset)
        
        # Load cc[i][j]
        cc_offset = i * LEN_2D + j
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute and store bb[i][j]
        result = bb_prev_val + cc_val
        bb_offset = i * LEN_2D + j
        tl.store(bb_ptr + bb_offset, result)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(128, triton.next_power_of_2(LEN_2D))
    
    grid = (LEN_2D - 1,)
    
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )