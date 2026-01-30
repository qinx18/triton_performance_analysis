import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    for j in range(1, len_2d):
        # Load aa[j-1][i] for all i in this block
        prev_aa_ptrs = aa_ptr + (j - 1) * len_2d + i_idx
        prev_aa_vals = tl.load(prev_aa_ptrs, mask=i_mask, other=0.0)
        
        # Load bb[j][i] for all i in this block
        bb_ptrs = bb_ptr + j * len_2d + i_idx
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        new_aa_vals = prev_aa_vals + bb_vals
        
        # Store aa[j][i] for all i in this block
        aa_ptrs = aa_ptr + j * len_2d + i_idx
        tl.store(aa_ptrs, new_aa_vals, mask=i_mask)

def s231_triton(aa, bb, len_2d):
    BLOCK_SIZE = 32
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )