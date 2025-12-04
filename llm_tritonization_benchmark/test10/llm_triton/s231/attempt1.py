import torch
import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] for all i in this block
        prev_j_offset = (j - 1) * LEN_2D + i_idx
        aa_prev = tl.load(aa_ptr + prev_j_offset, mask=i_mask, other=0.0)
        
        # Load bb[j][i] for all i in this block
        bb_offset = j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptr + bb_offset, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = aa_prev + bb_vals
        
        # Store to aa[j][i]
        curr_j_offset = j * LEN_2D + i_idx
        tl.store(aa_ptr + curr_j_offset, result, mask=i_mask)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa, bb, LEN_2D, BLOCK_SIZE
    )
    
    return aa