import torch
import triton
import triton.language as tl

@triton.jit
def s1119_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_j = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid_j * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(1, LEN_2D):
        # Load aa[i-1][j] for all j in this block
        prev_aa_ptrs = aa_ptr + (i - 1) * LEN_2D + j_idx
        prev_aa_vals = tl.load(prev_aa_ptrs, mask=j_mask, other=0.0)
        
        # Load bb[i][j] for all j in this block
        bb_ptrs = bb_ptr + i * LEN_2D + j_idx
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
        result = prev_aa_vals + bb_vals
        
        # Store aa[i][j] for all j in this block
        aa_ptrs = aa_ptr + i * LEN_2D + j_idx
        tl.store(aa_ptrs, result, mask=j_mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1119_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )