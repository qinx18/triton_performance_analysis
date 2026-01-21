import triton
import triton.language as tl

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    
    mask_i = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_prev_offsets = (j - 1) * LEN_2D + i_idx
        aa_curr_offsets = j * LEN_2D + i_idx
        cc_offsets = j * LEN_2D + i_idx
        
        aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=mask_i, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask_i, other=0.0)
        
        aa_new_vals = aa_prev_vals + cc_vals
        tl.store(aa_ptr + aa_curr_offsets, aa_new_vals, mask=mask_i)
        
        # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        bb_prev_offsets = (i_idx - 1) * LEN_2D + j
        bb_curr_offsets = i_idx * LEN_2D + j
        cc_offsets_2 = i_idx * LEN_2D + j
        
        bb_prev_vals = tl.load(bb_ptr + bb_prev_offsets, mask=mask_i, other=0.0)
        cc_vals_2 = tl.load(cc_ptr + cc_offsets_2, mask=mask_i, other=0.0)
        
        bb_new_vals = bb_prev_vals + cc_vals_2
        tl.store(bb_ptr + bb_curr_offsets, bb_new_vals, mask=mask_i)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s2233_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )