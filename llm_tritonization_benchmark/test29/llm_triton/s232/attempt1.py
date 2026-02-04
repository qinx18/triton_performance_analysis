import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(1, len_2d):
        valid_mask = j_mask & (j_idx >= 1) & (j_idx < len_2d) & (i <= j_idx)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            prev_aa_ptrs = aa_ptr + j_idx * len_2d + (i - 1)
            bb_ptrs = aa_ptr + j_idx * len_2d + i
            curr_bb_ptrs = bb_ptr + j_idx * len_2d + i
            
            prev_aa_vals = tl.load(prev_aa_ptrs, mask=valid_mask, other=0.0)
            bb_vals = tl.load(curr_bb_ptrs, mask=valid_mask, other=0.0)
            
            result = prev_aa_vals * prev_aa_vals + bb_vals
            
            tl.store(bb_ptrs, result, mask=valid_mask)

def s232_triton(aa, bb, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s232_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )