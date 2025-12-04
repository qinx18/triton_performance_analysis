import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D - 1
    
    for i in range(1, LEN_2D):
        # Load bb[i][j]
        bb_idx = i * LEN_2D + j_idx + 1
        bb_vals = tl.load(bb_ptr + bb_idx, mask=j_mask)
        
        # Load aa[i-1][j-1]
        aa_prev_idx = (i - 1) * LEN_2D + j_idx
        aa_prev_vals = tl.load(aa_ptr + aa_prev_idx, mask=j_mask)
        
        # Compute aa[i][j] = aa[i-1][j-1] + bb[i][j]
        result = aa_prev_vals + bb_vals
        
        # Store to aa[i][j]
        aa_curr_idx = i * LEN_2D + j_idx + 1
        tl.store(aa_ptr + aa_curr_idx, result, mask=j_mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s119_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )