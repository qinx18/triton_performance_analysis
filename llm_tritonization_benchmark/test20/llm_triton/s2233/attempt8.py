import triton
import triton.language as tl

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_indices < LEN_2D
    
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_prev_offsets = (j - 1) * LEN_2D + i_indices
        aa_curr_offsets = j * LEN_2D + i_indices
        cc_offsets = j * LEN_2D + i_indices
        
        aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=i_mask, other=0.0)
        aa_new_vals = aa_prev_vals + cc_vals
        tl.store(aa_ptr + aa_curr_offsets, aa_new_vals, mask=i_mask)
    
    # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
    for j in range(1, LEN_2D):
        bb_prev_offsets = (i_indices - 1) * LEN_2D + j
        bb_curr_offsets = i_indices * LEN_2D + j
        cc_offsets_bb = i_indices * LEN_2D + j
        
        valid_mask = i_mask
        bb_prev_vals = tl.load(bb_ptr + bb_prev_offsets, mask=valid_mask, other=0.0)
        cc_vals_bb = tl.load(cc_ptr + cc_offsets_bb, mask=valid_mask, other=0.0)
        bb_new_vals = bb_prev_vals + cc_vals_bb
        tl.store(bb_ptr + bb_curr_offsets, bb_new_vals, mask=valid_mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    s2233_kernel[grid](aa, bb, cc, LEN_2D, BLOCK_SIZE)