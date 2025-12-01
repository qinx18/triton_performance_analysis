import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get column index
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid columns
    i_mask = i < LEN_2D
    
    # Process each row sequentially due to data dependency
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] and bb[j][i]
        prev_row_offset = (j - 1) * LEN_2D + i
        curr_row_offset = j * LEN_2D + i
        
        aa_prev = tl.load(aa_ptr + prev_row_offset, mask=i_mask, other=0.0)
        bb_curr = tl.load(bb_ptr + curr_row_offset, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = aa_prev + bb_curr
        
        # Store result
        tl.store(aa_ptr + curr_row_offset, result, mask=i_mask)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )