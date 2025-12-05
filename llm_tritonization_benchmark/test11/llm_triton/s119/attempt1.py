import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D - 1
    j_valid = j_idx + 1
    
    for i in range(1, LEN_2D):
        # Read aa[i-1][j-1] for valid indices
        prev_i = i - 1
        prev_j = j_valid - 1
        aa_prev_offsets = prev_i * LEN_2D + prev_j
        aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=j_mask)
        
        # Read bb[i][j]
        bb_offsets = i * LEN_2D + j_valid
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute aa[i][j] = aa[i-1][j-1] + bb[i][j]
        result = aa_prev + bb_vals
        
        # Store aa[i][j]
        aa_offsets = i * LEN_2D + j_valid
        tl.store(aa_ptr + aa_offsets, result, mask=j_mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s119_kernel[grid](
        aa, bb, LEN_2D, BLOCK_SIZE
    )