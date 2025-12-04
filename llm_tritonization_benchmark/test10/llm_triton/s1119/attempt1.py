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
        # Load aa[i-1][j]
        prev_row_offset = (i - 1) * LEN_2D + j_idx
        aa_prev = tl.load(aa_ptr + prev_row_offset, mask=j_mask, other=0.0)
        
        # Load bb[i][j]
        curr_row_offset = i * LEN_2D + j_idx
        bb_curr = tl.load(bb_ptr + curr_row_offset, mask=j_mask, other=0.0)
        
        # Compute aa[i][j] = aa[i-1][j] + bb[i][j]
        result = aa_prev + bb_curr
        
        # Store aa[i][j]
        tl.store(aa_ptr + curr_row_offset, result, mask=j_mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.size(0)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1119_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa