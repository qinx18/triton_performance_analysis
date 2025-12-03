import torch
import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get column index
    col_idx = tl.program_id(0)
    
    if col_idx >= LEN_2D:
        return
    
    # Process column sequentially due to data dependency
    for j in range(1, LEN_2D):
        # Load aa[j-1][col_idx]
        prev_offset = (j - 1) * LEN_2D + col_idx
        prev_val = tl.load(aa_ptr + prev_offset)
        
        # Load bb[j][col_idx]
        bb_offset = j * LEN_2D + col_idx
        bb_val = tl.load(bb_ptr + bb_offset)
        
        # Compute and store aa[j][col_idx] = aa[j-1][col_idx] + bb[j][col_idx]
        result = prev_val + bb_val
        aa_offset = j * LEN_2D + col_idx
        tl.store(aa_ptr + aa_offset, result)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 1
    
    # Launch kernel with one thread per column
    grid = (LEN_2D,)
    
    s231_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )