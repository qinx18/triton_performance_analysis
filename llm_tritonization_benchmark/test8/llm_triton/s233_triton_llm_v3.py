import torch
import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        j_offsets = j_start + offsets
        mask = j_offsets < LEN_2D
        
        # Load aa[j-1][i] + cc[j][i]
        aa_prev_idx = (j_offsets - 1) * LEN_2D + i
        cc_idx = j_offsets * LEN_2D + i
        aa_current_idx = j_offsets * LEN_2D + i
        
        aa_prev = tl.load(aa_ptr + aa_prev_idx, mask=mask)
        cc_val = tl.load(cc_ptr + cc_idx, mask=mask)
        result = aa_prev + cc_val
        
        tl.store(aa_ptr + aa_current_idx, result, mask=mask)
    
    # Second loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for j_start in range(1, LEN_2D, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        j_offsets = j_start + offsets
        mask = j_offsets < LEN_2D
        
        # Load bb[j][i-1] + cc[j][i]
        bb_prev_idx = j_offsets * LEN_2D + (i - 1)
        cc_idx = j_offsets * LEN_2D + i
        bb_current_idx = j_offsets * LEN_2D + i
        
        bb_prev = tl.load(bb_ptr + bb_prev_idx, mask=mask)
        cc_val = tl.load(cc_ptr + cc_idx, mask=mask)
        result = bb_prev + cc_val
        
        tl.store(bb_ptr + bb_current_idx, result, mask=mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    grid = (LEN_2D - 1,)
    
    s233_kernel[grid](
        aa, bb, cc, 
        LEN_2D, BLOCK_SIZE
    )