import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] (previous row)
        prev_row_ptrs = aa_ptr + (j - 1) * LEN_2D + i_idx
        prev_vals = tl.load(prev_row_ptrs, mask=i_mask, other=0.0)
        
        # Load bb[j][i] (current row)
        bb_ptrs = bb_ptr + j * LEN_2D + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = prev_vals + bb_vals
        
        # Store aa[j][i]
        curr_row_ptrs = aa_ptr + j * LEN_2D + i_idx
        tl.store(curr_row_ptrs, result, mask=i_mask)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa, bb, LEN_2D, BLOCK_SIZE
    )