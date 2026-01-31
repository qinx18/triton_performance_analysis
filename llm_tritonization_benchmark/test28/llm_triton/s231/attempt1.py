import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa, bb, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < len_2d
    
    for j in range(1, len_2d):
        # Load aa[j-1][i] and bb[j][i]
        aa_prev_ptrs = aa + (j - 1) * len_2d + i_idx
        bb_curr_ptrs = bb + j * len_2d + i_idx
        
        aa_prev_vals = tl.load(aa_prev_ptrs, mask=i_mask, other=0.0)
        bb_curr_vals = tl.load(bb_curr_ptrs, mask=i_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = aa_prev_vals + bb_curr_vals
        
        # Store to aa[j][i]
        aa_curr_ptrs = aa + j * len_2d + i_idx
        tl.store(aa_curr_ptrs, result, mask=i_mask)

def s231_triton(aa, bb, len_2d):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s231_kernel[grid](
        aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )