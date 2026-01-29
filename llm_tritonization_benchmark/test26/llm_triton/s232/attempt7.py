import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D):
        # Only process j values where j >= i (triangular constraint)
        valid_mask = j_mask & (j_idx >= i)
        
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            # Load aa[j][i-1] 
            prev_offsets = j_idx * LEN_2D + (i - 1)
            prev_vals = tl.load(aa_ptr + prev_offsets, mask=valid_mask, other=0.0)
            
            # Load bb[j][i]
            bb_offsets = j_idx * LEN_2D + i
            bb_vals = tl.load(bb_ptr + bb_offsets, mask=valid_mask, other=0.0)
            
            # Compute aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i]
            result = prev_vals * prev_vals + bb_vals
            
            # Store result to aa[j][i]
            curr_offsets = j_idx * LEN_2D + i
            tl.store(aa_ptr + curr_offsets, result, mask=valid_mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s232_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )