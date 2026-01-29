import torch
import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        prev_j = j - 1
        
        # Load aa[j-1][i] and bb[j][i]
        prev_aa_ptrs = aa_ptr + prev_j * LEN_2D + i_idx
        bb_ptrs = bb_ptr + j * LEN_2D + i_idx
        curr_aa_ptrs = aa_ptr + j * LEN_2D + i_idx
        
        prev_aa_vals = tl.load(prev_aa_ptrs, mask=i_mask, other=0.0)
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = prev_aa_vals + bb_vals
        
        # Store result
        tl.store(curr_aa_ptrs, result, mask=i_mask)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa, bb, 
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )