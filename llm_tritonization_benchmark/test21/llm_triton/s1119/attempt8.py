import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D):
        # Calculate memory offsets for aa[i-1][j]
        aa_prev_offsets = (i - 1) * LEN_2D + j_idx
        # Calculate memory offsets for bb[i][j]
        bb_curr_offsets = i * LEN_2D + j_idx
        # Calculate memory offsets for aa[i][j]
        aa_curr_offsets = i * LEN_2D + j_idx
        
        # Load data
        aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=j_mask, other=0.0)
        bb_curr = tl.load(bb_ptr + bb_curr_offsets, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
        result = aa_prev + bb_curr
        
        # Store result
        tl.store(aa_ptr + aa_curr_offsets, result, mask=j_mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa