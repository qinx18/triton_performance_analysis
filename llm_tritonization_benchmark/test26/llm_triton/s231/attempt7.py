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
        # Load aa[j-1][i] for all i in this block
        prev_row_offset = (j - 1) * LEN_2D
        prev_aa_ptrs = aa_ptr + prev_row_offset + i_idx
        prev_aa_vals = tl.load(prev_aa_ptrs, mask=i_mask, other=0.0)
        
        # Load bb[j][i] for all i in this block
        curr_row_offset = j * LEN_2D
        bb_ptrs = bb_ptr + curr_row_offset + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        new_aa_vals = prev_aa_vals + bb_vals
        
        # Store aa[j][i] for all i in this block
        aa_ptrs = aa_ptr + curr_row_offset + i_idx
        tl.store(aa_ptrs, new_aa_vals, mask=i_mask)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa,
        bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )