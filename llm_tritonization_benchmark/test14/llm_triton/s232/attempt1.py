import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D + 1):
        i_mask = j_idx >= i
        combined_mask = j_mask & i_mask
        
        if i == 1:
            prev_idx = j_idx * LEN_2D + (i - 1)
            curr_idx = j_idx * LEN_2D + i
            bb_idx = j_idx * LEN_2D + i
            
            prev_vals = tl.load(aa_ptr + prev_idx, mask=combined_mask)
            bb_vals = tl.load(bb_ptr + bb_idx, mask=combined_mask)
            
            result = prev_vals * prev_vals + bb_vals
            tl.store(aa_ptr + curr_idx, result, mask=combined_mask)
        else:
            prev_idx = j_idx * LEN_2D + (i - 1)
            curr_idx = j_idx * LEN_2D + i
            bb_idx = j_idx * LEN_2D + i
            
            prev_vals = tl.load(aa_ptr + prev_idx, mask=combined_mask)
            bb_vals = tl.load(bb_ptr + bb_idx, mask=combined_mask)
            
            result = prev_vals * prev_vals + bb_vals
            tl.store(aa_ptr + curr_idx, result, mask=combined_mask)

def s232_triton(aa, bb):
    LEN_2D = aa.size(0)
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s232_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )