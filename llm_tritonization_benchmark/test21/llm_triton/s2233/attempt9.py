import triton
import triton.language as tl

@triton.jit
def s2233_kernel(aa, bb, cc, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_idx < LEN_2D
    
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_prev_offsets = (j - 1) * LEN_2D + i_idx
        aa_curr_offsets = j * LEN_2D + i_idx
        cc_offsets = j * LEN_2D + i_idx
        
        aa_prev_vals = tl.load(aa + aa_prev_offsets, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc + cc_offsets, mask=i_mask, other=0.0)
        result = aa_prev_vals + cc_vals
        tl.store(aa + aa_curr_offsets, result, mask=i_mask)
    
    for j in range(1, LEN_2D):
        # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        bb_prev_offsets = (i_idx - 1) * LEN_2D + j
        bb_curr_offsets = i_idx * LEN_2D + j
        cc_offsets = i_idx * LEN_2D + j
        
        bb_prev_vals = tl.load(bb + bb_prev_offsets, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc + cc_offsets, mask=i_mask, other=0.0)
        result = bb_prev_vals + cc_vals
        tl.store(bb + bb_curr_offsets, result, mask=i_mask)

def s2233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    
    s2233_kernel[grid](aa, bb, cc, LEN_2D, BLOCK_SIZE)