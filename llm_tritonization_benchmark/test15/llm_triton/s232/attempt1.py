import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D):
        i_mask = j_idx >= i
        mask = j_mask & i_mask
        
        # Load aa[j][i-1]
        prev_offsets = j_idx * LEN_2D + (i - 1)
        aa_prev = tl.load(aa_ptr + prev_offsets, mask=mask)
        
        # Load bb[j][i]
        curr_offsets = j_idx * LEN_2D + i
        bb_curr = tl.load(bb_ptr + curr_offsets, mask=mask)
        
        # Compute aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i]
        result = aa_prev * aa_prev + bb_curr
        
        # Store aa[j][i]
        tl.store(aa_ptr + curr_offsets, result, mask=mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s232_kernel[grid](
        aa, bb, 
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )