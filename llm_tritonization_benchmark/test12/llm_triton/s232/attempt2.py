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
        prev_ptrs = aa_ptr + j_idx * LEN_2D + (i - 1)
        aa_prev = tl.load(prev_ptrs, mask=mask, other=0.0)
        
        # Load bb[j][i]
        bb_ptrs = bb_ptr + j_idx * LEN_2D + i
        bb_val = tl.load(bb_ptrs, mask=mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i]
        result = aa_prev * aa_prev + bb_val
        
        # Store aa[j][i]
        aa_ptrs = aa_ptr + j_idx * LEN_2D + i
        tl.store(aa_ptrs, result, mask=mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s232_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )