import torch
import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE_I: tl.constexpr):
    # Get the column index (i dimension)
    i = tl.program_id(0) * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    i_mask = i < LEN_2D
    
    # Process each row sequentially due to data dependency
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] (previous row)
        prev_offsets = (j - 1) * LEN_2D + i
        aa_prev = tl.load(aa_ptr + prev_offsets, mask=i_mask, other=0.0)
        
        # Load bb[j][i] (current row)
        bb_offsets = j * LEN_2D + i
        bb_curr = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        aa_new = aa_prev + bb_curr
        
        # Store result to aa[j][i]
        aa_offsets = j * LEN_2D + i
        tl.store(aa_ptr + aa_offsets, aa_new, mask=i_mask)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE_I = min(256, triton.next_power_of_2(LEN_2D))
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
    
    s231_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE_I=BLOCK_SIZE_I
    )